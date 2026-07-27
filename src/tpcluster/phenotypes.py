from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str:
    lookup = {_normalise(column): str(column) for column in columns}
    for alias in aliases:
        match = lookup.get(_normalise(alias))
        if match is not None