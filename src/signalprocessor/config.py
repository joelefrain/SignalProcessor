from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path.
    import tomli as tomllib  # type: ignore


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".toml", ".tml"}:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def as_period_range(value: Any) -> tuple[float, float] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, dict):
        return float(value["min"]), float(value["max"])
    lo, hi = value
    return float(lo), float(hi)
