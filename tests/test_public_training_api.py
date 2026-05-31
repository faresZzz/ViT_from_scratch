import importlib
import inspect
import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_AND_NOTEBOOK_TARGETS = [
    PROJECT_ROOT / "README.md",
    *sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")),
    *sorted((PROJECT_ROOT / "tests").glob("test_*.py")),
]


def _import_or_fail(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised only on broken integration states
        pytest.fail(
            f"Expected `{module_name}` to import cleanly for the public training API. "
            f"Current error: {exc!r}"
        )


def test_training_module_exposes_train_without_requiring_fit():
    training_module = _import_or_fail("vit_from_scratch.training")

    assert hasattr(
        training_module, "train"
    ), "Missing `vit_from_scratch.training.train`; the public multi-epoch API must be `train`."
    assert callable(
        training_module.train
    ), "`vit_from_scratch.training.train` must be callable."


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("train", "callable"),
        ("classification", "module"),
        ("mae", "module"),
        ("dino", "module"),
        ("create_experiment_run", "callable"),
        ("save_checkpoint", "callable"),
        ("load_history", "callable"),
        ("train_classification", "callable"),
        ("train_masked_autoencoder", "callable"),
        ("train_dino", "callable"),
    ],
)
def test_root_package_re_exports_training_surface(name: str, kind: str):
    package = _import_or_fail("vit_from_scratch")

    assert hasattr(
        package, name
    ), f"Missing root export `{name}`; expected `from vit_from_scratch import {name}` to work."

    exported = getattr(package, name)
    if kind == "callable":
        assert callable(exported), f"`vit_from_scratch.{name}` must be callable."
    else:
        assert inspect.ismodule(exported) or hasattr(
            exported, "train"
        ), f"`vit_from_scratch.{name}` should expose the corresponding training module or method entrypoint."


def test_user_facing_files_do_not_advertise_fit_as_the_main_api():
    legacy_name = "fi" "t"
    fit_call_pattern = re.compile(rf"(?<!\w){re.escape(legacy_name)}\s*\(")
    fit_import_pattern = re.compile(
        rf"from\s+vit_from_scratch\s+import\s+{re.escape(legacy_name)}\b"
    )
    violations: list[str] = []

    for path in README_AND_NOTEBOOK_TARGETS:
        content = path.read_text(encoding="utf-8")
        if fit_call_pattern.search(content):
            violations.append(
                f"{path.relative_to(PROJECT_ROOT)} contains the legacy training call syntax."
            )
        if fit_import_pattern.search(content):
            violations.append(
                f"{path.relative_to(PROJECT_ROOT)} contains the legacy root import syntax."
            )

    assert not violations, "User-facing legacy training usage still present:\n- " + "\n- ".join(
        violations
    )
