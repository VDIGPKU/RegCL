import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "datasets"


def resolve_dataset_path(path: str, repo_root: Path = REPO_ROOT) -> str:
    """Resolve dataset paths from config files.

    Absolute paths are kept unchanged. Relative paths are interpreted from the
    repository root so scripts can be launched from different working directories.
    """
    expanded = Path(os.path.expandvars(os.path.expanduser(path)))
    if expanded.is_absolute():
        return str(expanded)
    return str(Path(repo_root) / expanded)


def load_dataset_config(config_path: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        key: resolve_dataset_path(value, repo_root)
        if isinstance(value, str)
        else value
        for key, value in data.items()
    }


def select_dataset_order(
    train_data: Dict[str, Any],
    test_data: Dict[str, Any],
    order: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    keys = order.split("_")
    missing_keys = [
        key for key in keys
        if key not in train_data or key not in test_data
    ]
    if missing_keys:
        raise KeyError(f"Missing dataset keys in config: {missing_keys}")
    return (
        {key: train_data[key] for key in keys},
        {key: test_data[key] for key in keys},
    )


def load_train_test_configs(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    order: str = "Kvasir_camo_ISTD_ISIC_cod",
    repo_root: Path = REPO_ROOT,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config_dir = Path(config_dir)
    train_data = load_dataset_config(config_dir / "datasets_train.json", repo_root)
    test_data = load_dataset_config(config_dir / "datasets_test.json", repo_root)
    return select_dataset_order(train_data, test_data, order)
