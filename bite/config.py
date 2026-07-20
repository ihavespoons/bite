"""Minimal YAML config loader with single-level ``defaults`` merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config, merging a referenced ``defaults:`` file (resolved relative to it)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("defaults", None)
    if parent:
        base = load_config(path.parent / parent)
        return _deep_merge(base, data)
    return data
