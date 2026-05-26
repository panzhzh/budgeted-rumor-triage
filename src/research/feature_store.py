from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import ensure_dir


def normalize_text_for_hash(text: str | None) -> str:
    return " ".join((text or "").split()).strip().lower()


def text_hash(text: str | None) -> str:
    normalized = normalize_text_for_hash(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def config_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def read_parquet_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def merge_unique_rows(
    existing: pd.DataFrame,
    additions: pd.DataFrame,
    *,
    key: str,
) -> pd.DataFrame:
    if existing.empty:
        if additions.empty:
            return additions.copy()
        return additions.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    if additions.empty:
        return existing.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    merged = pd.concat([existing, additions], ignore_index=True)
    merged = merged.drop_duplicates(subset=[key], keep="last")
    return merged.reset_index(drop=True)


def write_frame(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    csv_path: Path | None = None,
) -> None:
    ensure_dir(parquet_path.parent)
    frame.to_parquet(parquet_path, index=False)
    if csv_path is not None:
        ensure_dir(csv_path.parent)
        frame.to_csv(csv_path, index=False)

