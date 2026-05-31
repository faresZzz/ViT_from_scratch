import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
README_PATH = PROJECT_ROOT / "README.md"


def _iter_notebooks() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _read_notebook(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _combined_notebook_source(path: Path) -> str:
    notebook = _read_notebook(path)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_notebooks_are_output_free_and_restartable():
    failures: list[str] = []

    for notebook_path in _iter_notebooks():
        notebook = _read_notebook(notebook_path)
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue

            outputs = cell.get("outputs", [])
            if outputs:
                failures.append(
                    f"{notebook_path.name} cell {index} stores {len(outputs)} output(s)."
                )

            if cell.get("execution_count", None) is not None:
                failures.append(
                    f"{notebook_path.name} cell {index} has execution_count="
                    f"{cell.get('execution_count')!r} instead of null."
                )

    assert not failures, "Notebook cleanliness failures:\n- " + "\n- ".join(failures)


def test_notebooks_do_not_patch_sys_path():
    offenders: list[str] = []

    for notebook_path in _iter_notebooks():
        notebook = _read_notebook(notebook_path)
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            source = "".join(cell.get("source", []))
            if "sys.path" in source:
                offenders.append(
                    f"{notebook_path.name} cell {index} contains forbidden sys.path usage."
                )

    assert not offenders, "Notebook sys.path violations:\n- " + "\n- ".join(offenders)


def test_readme_and_notebooks_do_not_reference_local_absolute_paths():
    forbidden_tokens = ("/Users/", "file://", "vscode://")

    readme_text = README_PATH.read_text(encoding="utf-8")
    for token in forbidden_tokens:
        assert token not in readme_text, f"README.md should not contain {token!r}"

    offenders: list[str] = []
    for notebook_path in _iter_notebooks():
        combined_source = _combined_notebook_source(notebook_path)
        for token in forbidden_tokens:
            if token in combined_source:
                offenders.append(f"{notebook_path.name} contains forbidden token {token!r}.")

    assert not offenders, "Notebook path violations:\n- " + "\n- ".join(offenders)


def test_experiment_results_notebook_mentions_external_images_directory():
    combined_source = _combined_notebook_source(NOTEBOOK_DIR / "experiment_results.ipynb")

    assert "data/external_images/" in combined_source


def test_experiment_results_notebook_mentions_resume_metadata_fields():
    combined_source = _combined_notebook_source(NOTEBOOK_DIR / "experiment_results.ipynb")

    assert "resume_from" in combined_source
    assert "resumed_from_epoch" in combined_source
    assert "target_epochs" in combined_source


def test_experiment_results_notebook_describes_scheduler_without_outdated_claim():
    combined_source = _combined_notebook_source(NOTEBOOK_DIR / "experiment_results.ipynb")
    normalized = combined_source.lower()

    assert "scheduler" in normalized
    assert "sans early stopping ni scheduler" not in normalized
