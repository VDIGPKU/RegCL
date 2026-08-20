import argparse
import copy
import logging
import os
import re
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from merging.task_vectors import TaskVector


@dataclass
class DOPConfig:
    lr: float = 1e-4
    num_steps: int = 200
    mgda: bool = True
    ema: bool = True
    ema_beta: float = 0.99
    alpha: float = 0.5
    svd_epsilon: float = 1.0
    svd_proj_space: str = "uv"
    device: str = "auto"
    include_regex: list[str] = field(default_factory=list)
    exclude_regex: list[str] = field(default_factory=list)
    optimize_all_2d: bool = False
    show_progress: bool = True

    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not 0 <= self.svd_epsilon <= 1:
            raise ValueError("svd_epsilon must be in [0, 1]")
        if self.svd_proj_space not in ("u", "v", "uv"):
            raise ValueError("svd_proj_space must be one of: u, v, uv")


def load_checkpoint(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_tensors(checkpoint: Any) -> OrderedDict[str, Tensor]:
    tensors: OrderedDict[str, Tensor] = OrderedDict()
    if isinstance(checkpoint, Mapping):
        for key, value in sorted(checkpoint.items()):
            if isinstance(value, nn.Module):
                for sub_key, tensor in sorted(value.state_dict().items()):
                    if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
                        tensors[f"{key}.{sub_key}"] = tensor.detach().clone().cpu()
            elif torch.is_tensor(value) and torch.is_floating_point(value):
                tensors[str(key)] = value.detach().clone().cpu()
    elif isinstance(checkpoint, nn.Module):
        for key, tensor in sorted(checkpoint.state_dict().items()):
            if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
                tensors[key] = tensor.detach().clone().cpu()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return tensors


def checkpoint_from_tensors(
    reference_checkpoint: Any, tensors: Mapping[str, Tensor]
) -> Any:
    checkpoint = copy.deepcopy(reference_checkpoint)

    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
        state_dict.update({key: tensor.detach().clone() for key, tensor in tensors.items()})
        checkpoint.load_state_dict(state_dict, strict=True)
        return checkpoint

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    module_tensors: dict[str, dict[str, Tensor]] = {}
    for key, tensor in tensors.items():
        name, dot, sub_key = key.partition(".")
        if dot and isinstance(checkpoint.get(name), nn.Module):
            module_tensors.setdefault(name, {})[sub_key] = tensor.detach().clone()
        else:
            checkpoint[key] = tensor.detach().clone()

    for name, updates in module_tensors.items():
        state_dict = checkpoint[name].state_dict()
        state_dict.update(updates)
        checkpoint[name].load_state_dict(state_dict, strict=True)

    return checkpoint


def collect_base_model_paths(
    base_path: str,
    task_names: Iterable[str],
    init_name: str = "init.pth",
    ft_prefix: str = "ft_",
    meth: Optional[str] = None,
) -> tuple[str, list[str]]:
    init_path = os.path.join(base_path, init_name)
    ft_paths = [
        os.path.join(
            base_path,
            checkpoint_name(task_name, meth) if meth is not None else f"{ft_prefix}{task_name}.pth",
        )
        for task_name in task_names
    ]
    missing = [path for path in [init_path] + ft_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))
    return init_path, ft_paths


def checkpoint_name(task_name: str, meth: str) -> str:
    if meth == "sequ":
        return f"ft_{task_name}.pth"
    if meth == "indi":
        return f"{task_name}.pth"
    raise ValueError(f"Unsupported training method: {meth}")


def build_base_path(args: argparse.Namespace) -> str:
    base_dir = f"log/Comparison/{args.module}/magmax/{args.order}"
    return f"{base_dir}/{args.seed}_SEED"


def resolve_checkpoint_path(
    checkpoint: str,
    repo_root: str = REPO_ROOT,
    script_dir: str = SCRIPT_DIR,
) -> str:
    if os.path.isabs(checkpoint):
        candidates = [checkpoint]
    else:
        candidates = [
            os.path.abspath(checkpoint),
            os.path.abspath(os.path.join(repo_root, checkpoint)),
            os.path.abspath(os.path.join(script_dir, checkpoint)),
        ]

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "SAM checkpoint not found. Tried:\n" + "\n".join(unique_candidates)
    )


def setup_distribution(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    if args.cuda == -1 and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        logging.info(
            f"Initializing rank {args.rank} in "
            f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )
        dist.barrier()
    else:
        if args.cuda == -1:
            args.cuda = 0
        torch.cuda.set_device(args.cuda)
        args.rank = 0
        args.world_size = 1
        logging.info(f"Running in single GPU mode on cuda:{args.cuda}")


def build_merge_logger(base_path: str, merge_name: str):
    log_dir = os.path.join(base_path, merge_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"dop_merging.{merge_name}.{base_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode="w")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def merged_path(base_path: str, task_names: list[str], method: str = "dop") -> str:
    filename = f"init_{'_'.join(task_names)}.pth"
    return os.path.join(base_path, method, filename)


def discover_task_names(base_path: str, ft_prefix: str = "ft_") -> list[str]:
    names = []
    for filename in sorted(os.listdir(base_path)):
        if filename.startswith(ft_prefix) and filename.endswith(".pth"):
            names.append(filename[len(ft_prefix) : -len(".pth")])
    if not names:
        raise FileNotFoundError(f"No {ft_prefix}*.pth files found in {base_path}")
    return names


def task_names_from_order(order: Optional[str], base_path: str) -> list[str]:
    if order:
        return [name for name in order.split("_") if name]
    return discover_task_names(base_path)


def task_vector_from_tensors(
    init_tensors: Mapping[str, Tensor], finetuned_tensors: Mapping[str, Tensor]
) -> TaskVector:
    _ensure_matching_keys(init_tensors, finetuned_tensors, "finetuned tensors")
    return TaskVector(
        vector=OrderedDict(
            (key, finetuned_tensors[key] - init_tensors[key]) for key in init_tensors
        )
    )


def should_optimize_with_dop(key: str, tensor: Tensor, config: DOPConfig) -> bool:
    if not torch.is_floating_point(tensor) or tensor.dim() != 2:
        return False
    if config.include_regex and not _matches_any(key, config.include_regex):
        return False
    if config.exclude_regex and _matches_any(key, config.exclude_regex):
        return False
    return config.optimize_all_2d or key.endswith(".weight")


def merge_tensors_dop(
    init_tensors: Mapping[str, Tensor],
    finetuned_tensors_list: list[Mapping[str, Tensor]],
    config: DOPConfig,
) -> OrderedDict[str, Tensor]:
    if not finetuned_tensors_list:
        raise ValueError("finetuned_tensors_list must not be empty")

    init_tensors = OrderedDict((key, value.detach().clone().cpu()) for key, value in init_tensors.items())
    finetuned_tensors_list = [
        OrderedDict((key, value.detach().clone().cpu()) for key, value in tensors.items())
        for tensors in finetuned_tensors_list
    ]

    for index, tensors in enumerate(finetuned_tensors_list):
        _ensure_matching_keys(init_tensors, tensors, f"finetuned_tensors_list[{index}]")

    merged_tensors = OrderedDict(
        (key, value.detach().clone()) for key, value in finetuned_tensors_list[0].items()
    )

    for model_idx, next_tensors in enumerate(finetuned_tensors_list[1:], start=1):
        merged_tensors = merge_pair_tensors_dop(
            init_tensors,
            merged_tensors,
            next_tensors,
            config,
            model_idx=model_idx,
        )

    return merged_tensors


def merge_pair_tensors_dop(
    init_tensors: Mapping[str, Tensor],
    current_tensors: Mapping[str, Tensor],
    next_tensors: Mapping[str, Tensor],
    config: DOPConfig,
    model_idx: int = 0,
) -> OrderedDict[str, Tensor]:
    current_task_vector = task_vector_from_tensors(init_tensors, current_tensors)
    next_task_vector = task_vector_from_tensors(init_tensors, next_tensors)

    device = _resolve_device(config.device)
    merged: OrderedDict[str, Tensor] = OrderedDict()
    iterator = init_tensors.keys()
    if config.show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"DOP merging model {model_idx}")
        except ImportError:
            pass

    for key in iterator:
        pretrained = init_tensors[key]
        current = pretrained + current_task_vector.vector[key]
        new = pretrained + next_task_vector.vector[key]

        if should_optimize_with_dop(key, pretrained, config):
            merged[key] = optimize_weight_dop(
                pretrained,
                [current, new],
                config,
                key=key,
                device=device,
            )
        else:
            merged[key] = simple_average([current, new])

    return merged


def optimize_weight_dop(
    pretrained_weight: Tensor,
    finetuned_weights: list[Tensor],
    config: DOPConfig,
    key: str = "",
    device: Optional[torch.device] = None,
) -> Tensor:
    if len(finetuned_weights) != 2:
        raise ValueError("DOP checkpoint merging expects pairwise finetuned weights")
    if device is None:
        device = _resolve_device(config.device)

    original_dtype = pretrained_weight.dtype
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    pretrained = pretrained_weight.detach().to(device=device, dtype=work_dtype)
    finetuned = [
        weight.detach().to(device=device, dtype=work_dtype) for weight in finetuned_weights
    ]

    merged_weight = nn.Parameter(simple_average(finetuned).to(device), requires_grad=True)
    projections = [
        _task_vector_projection(weight - pretrained, config.svd_epsilon)
        for weight in finetuned
    ]

    if config.num_steps <= 0:
        return merged_weight.detach().to(dtype=original_dtype, device="cpu")

    optimizer = torch.optim.Adam([merged_weight], lr=config.lr)
    step_iter = range(config.num_steps)
    if config.show_progress:
        try:
            from tqdm import tqdm

            step_iter = tqdm(step_iter, desc=f"Optimizing {key}")
        except ImportError:
            pass

    ema_sol = [config.alpha, 1 - config.alpha]
    for _ in step_iter:
        if config.mgda:
            losses: dict[int, float] = {}
            grads: dict[int, Tensor] = {}
            for index, weight in enumerate(finetuned):
                loss_i = projection_loss(
                    merged_weight - weight, projections[index], config.svd_proj_space
                )
                losses[index] = float(loss_i.detach().cpu())
                optimizer.zero_grad()
                loss_i.backward()
                grads[index] = merged_weight.grad.detach().clone()

            for index in grads:
                normalizer = losses[index] if abs(losses[index]) > 1e-12 else 1.0
                grads[index] = grads[index] / float(normalizer)

            sol = _min_norm_solution_for_two(grads[0], grads[1])
            if config.ema:
                ema_sol = [
                    config.ema_beta * ema_sol[index]
                    + (1 - config.ema_beta) * float(sol[index])
                    for index in range(2)
                ]
                sol = ema_sol

            loss = merged_weight.new_tensor(0.0)
            for index, weight in enumerate(finetuned):
                loss_i = projection_loss(
                    merged_weight - weight, projections[index], config.svd_proj_space
                )
                loss = loss + float(sol[index]) * loss_i
        else:
            loss = merged_weight.new_tensor(0.0)
            for index, weight in enumerate(finetuned):
                loss_i = projection_loss(
                    merged_weight - weight, projections[index], config.svd_proj_space
                )
                scale = config.alpha if index == 0 else (1 - config.alpha)
                loss = loss + scale * loss_i

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return merged_weight.detach().to(dtype=original_dtype, device="cpu")


def projection_loss(
    delta_tv: Tensor,
    projection: tuple[Tensor, Tensor, Tensor],
    svd_proj_space: str,
) -> Tensor:
    proj_u, proj_s, proj_v = projection
    if proj_s.numel() == 0:
        return (delta_tv * 0).sum()

    diag_s = torch.diag(proj_s)
    proj_delta_u = diag_s @ proj_u.transpose(0, 1) @ delta_tv
    proj_delta_v = delta_tv @ proj_v @ diag_s
    loss_u = (proj_delta_u * proj_delta_u).sum()
    loss_v = (proj_delta_v * proj_delta_v).sum()

    if svd_proj_space == "uv":
        return loss_u + loss_v
    if svd_proj_space == "u":
        return loss_u
    if svd_proj_space == "v":
        return loss_v
    raise ValueError(f"Invalid svd_proj_space: {svd_proj_space}")


def simple_average(tensors: list[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("tensors must not be empty")
    result = tensors[0].detach().clone()
    for tensor in tensors[1:]:
        result = result + tensor.to(result.device)
    return result / len(tensors)


def save_dop_merge(
    init_path: str,
    ft_paths: list[str],
    task_names: list[str],
    base_path: str,
    config: DOPConfig,
    output_dir_name: str = "dop",
) -> str:
    return save_incremental_dop_merge(
        init_path,
        ft_paths,
        task_names,
        base_path,
        config,
        output_dir_name=output_dir_name,
    )


def save_incremental_dop_merge(
    init_path: str,
    ft_paths: list[str],
    task_names: list[str],
    base_path: str,
    config: DOPConfig,
    output_dir_name: str = "dop",
    merge_logger=None,
) -> str:
    init_checkpoint = load_checkpoint(init_path)
    init_tensors = checkpoint_tensors(init_checkpoint)
    ft_tensors = [checkpoint_tensors(load_checkpoint(path)) for path in ft_paths]
    merged_tensors = merge_tensors_dop(init_tensors, ft_tensors, config)

    save_path = merged_path(base_path, task_names, output_dir_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint_from_tensors(init_checkpoint, merged_tensors), save_path)
    if merge_logger is not None:
        merge_logger.info(f"Saved DOP checkpoint: {save_path}")
    return save_path


def build_dop_config_from_args(args: argparse.Namespace) -> DOPConfig:
    return DOPConfig(
        lr=args.lr,
        num_steps=args.num_steps,
        mgda=not args.no_mgda,
        ema=not args.no_ema,
        ema_beta=args.ema_beta,
        alpha=args.dop_alpha,
        svd_epsilon=args.svd_epsilon,
        svd_proj_space=args.svd_proj_space,
        device=args.dop_device,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        optimize_all_2d=args.all_2d,
        show_progress=not args.quiet,
    )


def run_merging(args, train_data, test_data, base_path: str):
    import torch.distributed as dist
    from importlib import import_module
    from segment_anything import sam_model_registry

    import train_regcl as train_regcl_lib
    from train_regcl import get_dataloader_SAM
    from merging.magmax_merging import validate_checkpoint

    train_regcl_lib.args = args
    task_names = list(train_data.keys())
    id_to_key = {index: key for index, key in enumerate(task_names)}
    init_path, ft_paths = collect_base_model_paths(
        base_path,
        task_names,
        init_name=args.init_name,
        ft_prefix=args.ft_prefix,
        meth=args.meth,
    )
    dop_config = build_dop_config_from_args(args)
    merge_method = args.output_dir_name

    lora_layer = list(range(args.begin, args.end + 1))
    ckpt_path = resolve_checkpoint_path(args.ckpt)
    logging.info(f"SAM checkpoint: {ckpt_path}")
    sam, img_embedding_size = sam_model_registry[args.vit_name](checkpoint=ckpt_path)
    sam = sam.cuda()
    for _, param in sam.image_encoder.named_parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)

    pkg = import_module(f"module.{args.module}")
    logging.info(f"Module {args.module} loaded to layers: {lora_layer}")
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam), lora_layer=lora_layer)

    testloader = {}
    for task_id, key in enumerate(task_names):
        train_data_location = train_data[key]
        test_data_location = test_data[key]
        if args.cuda == -1:
            _, testloader[task_id], _, _ = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )
        else:
            _, testloader[task_id] = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )

    merge_logger = logging
    if args.rank == 0:
        merge_logger = build_merge_logger(base_path, merge_method)
        merge_logger.info(str(args))
        merge_logger.info(f"base_path: {base_path}")

    AIJ = {}
    for task_id in range(len(task_names)):
        current_task_names = task_names[: task_id + 1]
        current_ft_paths = ft_paths[: task_id + 1]
        merge_path = merged_path(base_path, current_task_names, merge_method)
        if args.rank == 0:
            save_incremental_dop_merge(
                init_path,
                current_ft_paths,
                current_task_names,
                base_path,
                dop_config,
                output_dir_name=merge_method,
                merge_logger=merge_logger,
            )
        if args.cuda == -1:
            dist.barrier()

        if args.cuda != -1 or args.rank == 0:
            merge_logger.info(f"---Validation on {merge_path}---")
        results = validate_checkpoint(
            args,
            sam,
            net_origin,
            merge_path,
            testloader,
            task_id,
            id_to_key,
            merge_logger,
        )
        for index, scores in results.items():
            AIJ[(task_id, index)] = scores
        if args.rank == 0 or args.cuda != -1:
            train_regcl_lib.get_forgetting_metric(task_id, AIJ, merge_logger)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOP checkpoint merging and evaluation for AugModule LoRA checkpoints."
    )
    parser.add_argument(
        "--base_path",
        default=None,
        help="Folder containing init.pth and ft_<task>.pth checkpoints.",
    )
    parser.add_argument("--module", type=str, default="AugModule", choices=["AugModule", "LoRA"], help="Module")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--layers", type=str, default="7-11")
    parser.add_argument("--vit_name", type=str, default="vit_b")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(REPO_ROOT, "checkpoint", "sam_vit_b_01ec64.pth"),
        help="Pretrained SAM checkpoint",
    )
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--order",
        default="Kvasir_camo_ISTD_ISIC_cod",
        help="Task order such as Kvasir_camo_ISTD_ISIC_cod. If omitted, ft_*.pth files are sorted by name.",
    )
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument(
        "--meth",
        type=str,
        default="sequ",
        choices=["indi", "sequ"],
        help="Base checkpoint naming mode: indi uses task.pth; sequ uses ft_task.pth.",
    )
    parser.add_argument("--init-name", default="init.pth")
    parser.add_argument("--ft-prefix", default="ft_")
    parser.add_argument("--output-dir-name", default="dop")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--dop-alpha", "--alpha", dest="dop_alpha", type=float, default=0.5)
    parser.add_argument("--ema-beta", type=float, default=0.99)
    parser.add_argument("--svd-epsilon", type=float, default=1.0)
    parser.add_argument("--svd-proj-space", choices=("u", "v", "uv"), default="uv")
    parser.add_argument("--dop-device", "--device", dest="dop_device", default="auto")
    parser.add_argument("--include-regex", action="append", default=[])
    parser.add_argument("--exclude-regex", action="append", default=[])
    parser.add_argument("--all-2d", action="store_true")
    parser.add_argument("--no-mgda", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    args.alpha = 1
    args.beta = 10
    args.deta = 1
    args.begin = int(args.layers.split("-")[0])
    args.end = int(args.layers.split("-")[1])

    from train_regcl import load_dataset, setup_seed

    setup_seed(args.seed)

    base_path = args.base_path or build_base_path(args)
    os.makedirs(base_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    setup_distribution(args)
    logging.info(str(args))
    logging.info(f"base_path: {base_path}")
    train_data, test_data = load_dataset(args)
    run_merging(args, train_data, test_data, base_path)


def _task_vector_projection(
    task_vector: Tensor, svd_epsilon: float
) -> tuple[Tensor, Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(task_vector, full_matrices=True)
    v = vh.transpose(0, 1)
    if s.numel() == 0 or float(s.sum().detach().cpu()) == 0.0:
        return u[:, :0], s[:0], v[:, :0]

    cumsum_ratio = s.cumsum(dim=0) / s.sum()
    split_rank = torch.searchsorted(cumsum_ratio, svd_epsilon).item()
    return u[:, :split_rank], s[:split_rank], v[:, :split_rank]


def _min_norm_solution_for_two(grad_1: Tensor, grad_2: Tensor) -> list[float]:
    v1v1 = torch.mul(grad_1, grad_1).sum().detach().cpu()
    v1v2 = torch.mul(grad_1, grad_2).sum().detach().cpu()
    v2v2 = torch.mul(grad_2, grad_2).sum().detach().cpu()

    if v1v2 >= v1v1:
        gamma = 0.999
    elif v1v2 >= v2v2:
        gamma = 0.001
    else:
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        gamma = float(gamma)
    return [float(gamma), float(1 - gamma)]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(re.match(pattern, value) for pattern in patterns)


def _ensure_matching_keys(
    expected: Mapping[str, Tensor], actual: Mapping[str, Tensor], actual_name: str
) -> None:
    expected_keys = set(expected)
    actual_keys = set(actual)
    if expected_keys == actual_keys:
        return

    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    details = []
    if missing:
        details.append("missing keys: " + ", ".join(missing[:10]))
    if extra:
        details.append("extra keys: " + ", ".join(extra[:10]))
    raise ValueError(f"{actual_name} keys mismatch ({'; '.join(details)})")


if __name__ == "__main__":
    main()
