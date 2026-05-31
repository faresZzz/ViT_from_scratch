"""Helpers for loading and merging training configuration sources."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_training_config(path: str | Path | None) -> dict[str, object]:
    """Load a YAML training config or return an empty config when omitted."""

    if path is None:
        return {}

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Training config top-level YAML document must be a mapping.")
    return dict(payload)


def merge_config(
    defaults: dict[str, object],
    file_config: dict[str, object],
    cli_overrides: dict[str, object],
) -> dict[str, object]:
    """Merge defaults, YAML config, and explicit CLI values by precedence."""

    merged = dict(defaults)
    merged.update(file_config)
    merged.update(cli_overrides)
    return merged
