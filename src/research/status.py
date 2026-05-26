from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import socket
from typing import Any

from .io_utils import dump_json, ensure_dir


def write_status(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    enriched = dict(payload)
    enriched.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    enriched.setdefault("pid", os.getpid())
    enriched.setdefault("hostname", socket.gethostname())
    dump_json(path, enriched)


def stage_status_payload(
    *,
    dataset_name: str,
    stage: str,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    seconds: float | None = None,
    device: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": dataset_name,
        "stage": stage,
        "status": status,
    }
    if started_at is not None:
        payload["started_at"] = started_at
    if finished_at is not None:
        payload["finished_at"] = finished_at
    if seconds is not None:
        payload["seconds"] = round(seconds, 3)
    if device is not None:
        payload["device"] = device
    if extra:
        payload.update(extra)
    return payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_message(log_path: Path | None, message: str) -> None:
    timestamp = utc_now_iso()
    line = f"[{timestamp}] {message}"
    if log_path is not None:
        ensure_dir(log_path.parent)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line, flush=True)
