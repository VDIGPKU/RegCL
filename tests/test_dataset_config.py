import json
from pathlib import Path
import tempfile
import unittest


from datasets.config import load_train_test_configs


def write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class DatasetConfigTest(unittest.TestCase):
    def test_load_train_test_configs_resolves_relative_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_dir = tmp_path / "datasets"
            repo_root = tmp_path / "repo"
            config_dir.mkdir()
            repo_root.mkdir()
            write_json(
                config_dir / "datasets_train.json",
                {
                    "Kvasir": "data/Kvasir-SEG/train",
                    "camo": "data/CAMO/train",
                },
            )
            write_json(
                config_dir / "datasets_test.json",
                {
                    "Kvasir": "data/Kvasir-SEG/test",
                    "camo": "data/CAMO/test",
                },
            )

            train_data, test_data = load_train_test_configs(
                config_dir,
                order="camo_Kvasir",
                repo_root=repo_root,
            )

            self.assertEqual(list(train_data), ["camo", "Kvasir"])
            self.assertEqual(train_data["Kvasir"], str(repo_root / "data/Kvasir-SEG/train"))
            self.assertEqual(test_data["camo"], str(repo_root / "data/CAMO/test"))

    def test_load_train_test_configs_keeps_absolute_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_dir = tmp_path / "datasets"
            config_dir.mkdir()
            write_json(config_dir / "datasets_train.json", {"Kvasir": "/mnt/data/Kvasir/train"})
            write_json(config_dir / "datasets_test.json", {"Kvasir": "/mnt/data/Kvasir/test"})

            train_data, test_data = load_train_test_configs(config_dir, order="Kvasir", repo_root=tmp_path)

            self.assertEqual(train_data["Kvasir"], "/mnt/data/Kvasir/train")
            self.assertEqual(test_data["Kvasir"], "/mnt/data/Kvasir/test")

    def test_load_train_test_configs_reports_missing_order_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_dir = tmp_path / "datasets"
            config_dir.mkdir()
            write_json(config_dir / "datasets_train.json", {"Kvasir": "data/Kvasir-SEG/train"})
            write_json(config_dir / "datasets_test.json", {"Kvasir": "data/Kvasir-SEG/test"})

            with self.assertRaisesRegex(KeyError, "Missing dataset keys"):
                load_train_test_configs(config_dir, order="Kvasir_cod", repo_root=tmp_path)
