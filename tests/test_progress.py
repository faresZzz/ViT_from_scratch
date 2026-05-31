from __future__ import annotations

import pytest

from vit_from_scratch.progress import iter_progress


def test_iter_progress_disabled_yields_elements_without_wrapping():
    values = [1, 2, 3]

    result = list(
        iter_progress(
            values,
            desc="classification train 1/1",
            enabled=False,
        )
    )

    assert result == values


def test_iter_progress_enabled_calls_tqdm(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append({"iterable": iterable, **kwargs})
        return iterable

    monkeypatch.setattr("vit_from_scratch.progress.tqdm", fake_tqdm)

    values = [1, 2]
    result = list(iter_progress(values, desc="mae train 1/2", total=2))

    assert result == values
    assert calls == [
        {
            "iterable": values,
            "desc": "mae train 1/2",
            "total": 2,
        }
    ]
