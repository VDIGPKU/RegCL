import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class LayerStat:
    name: str
    layer_type: str
    is_linear: bool
    param_count: int
    total_ratio: float


@dataclass(frozen=True)
class ParamTotals:
    total_params: int
    linear_params: int
    non_linear_params: int

    @property
    def linear_ratio(self) -> float:
        if self.total_params == 0:
            return 0.0
        return self.linear_params / self.total_params

    @property
    def non_linear_ratio(self) -> float:
        if self.total_params == 0:
            return 0.0
        return self.non_linear_params / self.total_params


@dataclass(frozen=True)
class StatsReport:
    stats: list[LayerStat]
    totals: ParamTotals
    skipped: list[str]


def _direct_param_count(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters(recurse=False))


def _iter_parameter_layers(prefix: str, module: nn.Module) -> Iterable[tuple[str, nn.Module, int]]:
    param_count = _direct_param_count(module)
    if param_count > 0:
        yield prefix, module, param_count

    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        yield from _iter_parameter_layers(child_prefix, child)


def collect_layer_stats(checkpoint: Any, root_name: str = "model") -> list[LayerStat]:
    layers: list[tuple[str, nn.Module, int]] = []

    if isinstance(checkpoint, Mapping):
        for name, value in checkpoint.items():
            if isinstance(value, nn.Module):
                layers.extend(_iter_parameter_layers(str(name), value))
    elif isinstance(checkpoint, nn.Module):
        layers.extend(_iter_parameter_layers(root_name, checkpoint))

    total_params = sum(param_count for _, _, param_count in layers)
    return [
        LayerStat(
            name=name,
            layer_type=module.__class__.__name__,
            is_linear=isinstance(module, nn.Linear),
            param_count=param_count,
            total_ratio=(param_count / total_params) if total_params else 0.0,
        )
        for name, module, param_count in layers
    ]


def summarize_stats(stats: list[LayerStat]) -> ParamTotals:
    linear_params = sum(stat.param_count for stat in stats if stat.is_linear)
    non_linear_params = sum(stat.param_count for stat in stats if not stat.is_linear)
    return ParamTotals(
        total_params=linear_params + non_linear_params,
        linear_params=linear_params,
        non_linear_params=non_linear_params,
    )


def load_checkpoint(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_sam_model(path: str, model_type: str) -> tuple[nn.Module, int]:
    from segment_anything import sam_model_registry

    if model_type not in sam_model_registry:
        available = ", ".join(sorted(sam_model_registry))
        raise ValueError(f"Unsupported SAM model type: {model_type}. Available: {available}")
    return sam_model_registry[model_type](path)


def skipped_top_level_entries(checkpoint: Any) -> list[str]:
    if not isinstance(checkpoint, Mapping):
        return []
    return [
        str(name)
        for name, value in checkpoint.items()
        if not isinstance(value, nn.Module)
    ]


def select_sam_scope(sam: nn.Module, sam_scope: str) -> tuple[nn.Module, str]:
    if sam_scope == "all":
        return sam, "model"
    if sam_scope in ("image_encoder", "img_encoder"):
        return sam.image_encoder, "model.image_encoder"
    raise ValueError(f"Unsupported SAM scope: {sam_scope}")


def collect_checkpoint_stats(
    path: str,
    checkpoint_type: str = "adapter",
    model_type: str = "vit_b",
    sam_scope: str = "all",
) -> StatsReport:
    if checkpoint_type == "adapter":
        checkpoint = load_checkpoint(path)
        stats = collect_layer_stats(checkpoint)
        return StatsReport(
            stats=stats,
            totals=summarize_stats(stats),
            skipped=skipped_top_level_entries(checkpoint),
        )

    if checkpoint_type == "sam":
        sam, _ = load_sam_model(path, model_type)
        scoped_model, root_name = select_sam_scope(sam, sam_scope)
        stats = collect_layer_stats(scoped_model, root_name=root_name)
        return StatsReport(
            stats=stats,
            totals=summarize_stats(stats),
            skipped=[],
        )

    raise ValueError(f"Unsupported checkpoint type: {checkpoint_type}")


def format_ratio(ratio: float) -> str:
    return f"{ratio * 100:.4f}%"


def print_report(path: str, stats: list[LayerStat], totals: ParamTotals, skipped: list[str]) -> None:
    print(f"Checkpoint: {path}")
    print(f"Total counted parameters: {totals.total_params}")
    print(
        "Linear parameters: "
        f"{totals.linear_params} / {totals.total_params} ({format_ratio(totals.linear_ratio)})"
    )
    print(
        "Non-linear parameters: "
        f"{totals.non_linear_params} / {totals.total_params} ({format_ratio(totals.non_linear_ratio)})"
    )

    if skipped:
        print(f"Skipped non-module top-level entries: {', '.join(skipped)}")

    print()
    print(f"{'Layer':<36} {'Type':<16} {'Linear or not':<14} {'Params':>12} {'Total ratio':>12}")
    print("-" * 94)
    for stat in stats:
        linear_label = "linear" if stat.is_linear else "not linear"
        print(
            f"{stat.name:<36} "
            f"{stat.layer_type:<16} "
            f"{linear_label:<14} "
            f"{stat.param_count:>12} "
            f"{format_ratio(stat.total_ratio):>12}"
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count linear and non-linear parameter layers saved in one .pth/.pt checkpoint."
    )
    parser.add_argument("checkpoint", nargs="?", help="Path to the .pth/.pt checkpoint.")
    parser.add_argument("--path", dest="checkpoint_path", help="Path to the .pth/.pt checkpoint.")
    parser.add_argument(
        "--checkpoint-type",
        choices=("adapter", "sam"),
        default="adapter",
        help="Checkpoint format: adapter stores nn.Module objects; sam stores a SAM state_dict.",
    )
    parser.add_argument(
        "--model-type",
        choices=("vit_b", "vit_l", "vit_h"),
        default="vit_b",
        help="SAM model type used when --checkpoint-type sam.",
    )
    parser.add_argument(
        "--sam-scope",
        choices=("all", "image_encoder", "img_encoder"),
        default="all",
        help="SAM submodule to count when --checkpoint-type sam.",
    )
    args = parser.parse_args(argv)

    args.checkpoint = args.checkpoint_path or args.checkpoint
    if not args.checkpoint:
        parser.error("provide a checkpoint path, either as positional argument or with --path")
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    report = collect_checkpoint_stats(
        args.checkpoint,
        checkpoint_type=args.checkpoint_type,
        model_type=args.model_type,
        sam_scope=args.sam_scope,
    )
    print_report(args.checkpoint, report.stats, report.totals, report.skipped)


if __name__ == "__main__":
    main()
