import importlib.util
from pathlib import Path

import pytest
import yaml
from vit_from_scratch.train import _resolve_training_config, build_parser

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vit_from_scratch"
    / "training_config.py"
)
SPEC = importlib.util.spec_from_file_location("training_config", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
training_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_config)

load_training_config = training_config.load_training_config
merge_config = training_config.merge_config


def test_load_training_config_reads_yaml_mapping(tmp_path: Path):
    config_path = tmp_path / "training.yaml"
    config_path.write_text("epochs: 3\ndataset: fake\nshow_progress: false\n")

    config = load_training_config(config_path)

    assert config == {"epochs": 3, "dataset": "fake", "show_progress": False}


def test_load_training_config_returns_empty_dict_for_empty_file(tmp_path: Path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    assert load_training_config(config_path) == {}


def test_load_training_config_rejects_non_mapping_root(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- fake\n- config\n")

    with pytest.raises(ValueError, match="top-level YAML document must be a mapping"):
        load_training_config(config_path)


def test_merge_config_applies_defaults_then_yaml_then_cli():
    defaults = {"epochs": 5, "dataset": "cifar10", "show_progress": True}
    file_config = {"epochs": 3, "dataset": "fake"}
    cli_overrides = {"epochs": 1, "show_progress": False}

    merged = merge_config(defaults, file_config, cli_overrides)

    assert merged == {"epochs": 1, "dataset": "fake", "show_progress": False}


def test_resolve_training_config_parses_regularization_and_augmentation_fields(
    tmp_path: Path,
):
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: fake",
                "dropout: 0.1",
                "attention_dropout: 0.05",
                "label_smoothing: 0.02",
                "color_jitter_strength: 0.3",
                "random_erasing_prob: 0.15",
                "randaugment_num_ops: 2",
                "randaugment_magnitude: 7",
            ]
        )
    )

    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--dropout",
            "0.25",
            "--random-erasing-prob",
            "0.2",
        ]
    )

    resolved = _resolve_training_config(args)

    assert resolved["dropout"] == pytest.approx(0.25)
    assert resolved["attention_dropout"] == pytest.approx(0.05)
    assert resolved["label_smoothing"] == pytest.approx(0.02)
    assert resolved["color_jitter_strength"] == pytest.approx(0.3)
    assert resolved["random_erasing_prob"] == pytest.approx(0.2)
    assert resolved["randaugment_num_ops"] == 2
    assert resolved["randaugment_magnitude"] == 7


def test_resolve_training_config_parses_scheduler_and_split_fields(tmp_path: Path):
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: fake",
                "scheduler: cosine",
                "warmup_epochs: 5",
                "min_lr: 1.0e-5",
                "val_fraction: 0.2",
                "split_train_validation: true",
            ]
        )
    )

    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--warmup-epochs",
            "3",
            "--min-lr",
            "5.0e-6",
        ]
    )

    resolved = _resolve_training_config(args)

    assert resolved["scheduler"] == "cosine"
    assert resolved["warmup_epochs"] == 3
    assert resolved["min_lr"] == pytest.approx(5.0e-6)
    assert resolved["val_fraction"] == pytest.approx(0.2)
    assert resolved["split_train_validation"] is True


def test_training_yaml_configs_are_mappings_and_complete_for_real_dataset_runs():
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "configs" / "training"
    required_shared_keys = {
        "dataset",
        "epochs",
        "batch_size",
        "dropout",
        "attention_dropout",
        "label_smoothing",
        "color_jitter_strength",
        "random_erasing_prob",
        "randaugment_num_ops",
        "randaugment_magnitude",
        "device",
        "checkpoint_keep_last",
        "save_every",
        "scheduler",
        "warmup_epochs",
        "min_lr",
        "val_fraction",
        "split_train_validation",
    }
    real_dataset_complete_configs = {
        "classification_cifar10.yaml",
        "mae_cifar10.yaml",
        "dino_cifar10.yaml",
        "classification_stl10_small.yaml",
        "classification_stl10.yaml",
        "classification_tiny_imagenet.yaml",
        "dino_tiny_imagenet.yaml",
        "mae_stl10.yaml",
        "mae_tiny_imagenet.yaml",
        "dino_stl10.yaml",
    }
    required_complete_keys = {
        "dataset",
        "data_dir",
        "output_dir",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "image_size",
        "patch_size",
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "position_embedding",
        "device",
        "seed",
        "num_workers",
        "checkpoint_keep_last",
        "save_every",
        "show_progress",
        "scheduler",
        "warmup_epochs",
        "min_lr",
        "val_fraction",
        "split_train_validation",
        "dropout",
        "attention_dropout",
        "label_smoothing",
        "color_jitter_strength",
        "random_erasing_prob",
        "randaugment_num_ops",
        "randaugment_magnitude",
    }
    required_dino_view_keys = {
        "teacher_global_crops",
        "student_global_crops",
        "student_local_crops",
        "teacher_global_crop_scale_min",
        "teacher_global_crop_scale_max",
        "student_global_crop_scale_min",
        "student_global_crop_scale_max",
        "student_local_crop_scale_min",
        "student_local_crop_scale_max",
        "dino_view_noise_std",
    }
    expected_config_names = {
        "classification_stl10_small.yaml",
        "classification_tiny_imagenet.yaml",
        "mae_tiny_imagenet.yaml",
        "dino_tiny_imagenet.yaml",
    }

    existing_names = {path.name for path in config_dir.glob("*.yaml")}
    assert expected_config_names <= existing_names

    for config_path in sorted(config_dir.glob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert isinstance(payload, dict), f"{config_path.name} must contain a mapping"
        assert required_shared_keys <= payload.keys()
        assert payload["dataset"] in {"fake", "cifar10", "stl10", "tiny-imagenet"}
        assert payload["checkpoint_keep_last"] >= 1
        assert payload["save_every"] == 1
        assert payload["scheduler"] in {"none", "cosine"}
        assert payload["warmup_epochs"] >= 0
        assert payload["min_lr"] >= 0.0
        assert 0.0 < payload["val_fraction"] < 1.0
        assert isinstance(payload["split_train_validation"], bool)
        assert payload["dropout"] >= 0.0
        assert payload["attention_dropout"] >= 0.0
        assert payload["label_smoothing"] >= 0.0
        assert payload["color_jitter_strength"] >= 0.0
        assert payload["random_erasing_prob"] >= 0.0
        assert payload["randaugment_num_ops"] >= 0
        assert payload["randaugment_magnitude"] >= 0

        approach = payload.get("approach", "classification")
        assert approach in {"classification", "mae", "dino"}

        if config_path.name in real_dataset_complete_configs:
            assert required_complete_keys <= payload.keys()
            assert str(payload["output_dir"]).startswith("runs")

            if payload["dataset"] == "stl10":
                assert payload["image_size"] == 96
                assert payload["patch_size"] in {8, 12, 16}
                assert payload.get("external_image_dir") == "data/external_images"

            if payload["dataset"] == "tiny-imagenet":
                assert payload["image_size"] == 64
                assert payload["patch_size"] in {8, 16}
                assert payload["split_train_validation"] is True

            if approach == "mae":
                assert "mask_ratio" in payload
            if approach == "dino":
                assert {
                    "dino_momentum",
                    "student_temperature",
                    "teacher_temperature",
                } <= payload.keys()
                assert required_dino_view_keys <= payload.keys()
                assert payload["teacher_global_crops"] >= 1
                assert payload["student_global_crops"] >= 1
                assert payload["student_local_crops"] >= 0
                assert payload["student_global_crops"] >= payload["teacher_global_crops"]
                assert (
                    payload["teacher_global_crop_scale_min"]
                    < payload["teacher_global_crop_scale_max"]
                )
                assert (
                    payload["student_global_crop_scale_min"]
                    < payload["student_global_crop_scale_max"]
                )
                assert (
                    payload["student_local_crop_scale_min"]
                    < payload["student_local_crop_scale_max"]
                )
                assert payload["teacher_global_crop_scale_min"] > 0.0
                assert payload["student_global_crop_scale_min"] > 0.0
                assert payload["student_local_crop_scale_min"] > 0.0
                assert payload["teacher_global_crop_scale_max"] <= 1.0
                assert payload["student_global_crop_scale_max"] <= 1.0
                assert payload["student_local_crop_scale_max"] <= 1.0
                assert payload["dino_view_noise_std"] >= 0.0
        elif approach == "dino":
            assert required_dino_view_keys <= payload.keys()
            assert payload["teacher_global_crops"] >= 1
            assert payload["student_global_crops"] >= payload["teacher_global_crops"]
            assert payload["student_local_crops"] >= 0
            assert (
                payload["teacher_global_crop_scale_min"]
                < payload["teacher_global_crop_scale_max"]
            )
            assert (
                payload["student_global_crop_scale_min"]
                < payload["student_global_crop_scale_max"]
            )
            assert (
                payload["student_local_crop_scale_min"]
                < payload["student_local_crop_scale_max"]
            )
            assert payload["teacher_global_crop_scale_min"] > 0.0
            assert payload["student_global_crop_scale_min"] > 0.0
            assert payload["student_local_crop_scale_min"] > 0.0
            assert payload["teacher_global_crop_scale_max"] <= 1.0
            assert payload["student_global_crop_scale_max"] <= 1.0
            assert payload["student_local_crop_scale_max"] <= 1.0
            assert payload["dino_view_noise_std"] >= 0.0
