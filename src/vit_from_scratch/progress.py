"""Progress helpers shared by training runners."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only before runtime deps are installed
    def tqdm(iterable, **_: object):
        return iterable


def iter_progress(
    iterable: Iterable,
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterable | Iterator:
    """Wrap an iterable with ``tqdm`` when progress output is enabled."""

    if not enabled:
        return iterable
    return tqdm(iterable, desc=desc, total=total)
