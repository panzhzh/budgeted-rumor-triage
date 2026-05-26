from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import csv
import json
from pathlib import Path
import re
from typing import Iterable
from zoneinfo import ZoneInfo

from .schema import CANONICAL_FIELDS, normalize_record


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r"[!?！？。.!]+")
COMMON_TIME_PATTERNS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
)
CHINESE_MONTH_DAY_PATTERN = re.compile(
    r"^\s*(?P<month>\d{1,2})月(?P<day>\d{1,2})日(?:\s+(?P<hour>\d{1,2}):(?P<minute>\d{2}))?\s*$"
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_writable_dir(path: Path) -> Path:
    ensure_dir(path)
    probe = path / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return path


def is_hidden_artifact(name: str) -> bool:
    return name == ".DS_Store" or name == "__MACOSX" or name.startswith("._")


def iter_json_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    files = []
    for path in sorted(directory.iterdir()):
        if is_hidden_artifact(path.name):
            continue
        if path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return files


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_lines(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def dump_json(path: Path, payload: dict | list) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CanonicalWriter:
    def __init__(self, jsonl_path: Path, csv_path: Path) -> None:
        ensure_dir(jsonl_path.parent)
        ensure_dir(csv_path.parent)
        self._jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        self._csv_handle = csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=CANONICAL_FIELDS)
        self._csv_writer.writeheader()

    def write(self, record: dict) -> None:
        normalized = normalize_record(record)
        self._jsonl_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        csv_row = dict(normalized)
        csv_row["metadata"] = json.dumps(csv_row["metadata"], ensure_ascii=False, sort_keys=True)
        self._csv_writer.writerow(csv_row)

    def close(self) -> None:
        self._jsonl_handle.close()
        self._csv_handle.close()

    def __enter__(self) -> "CanonicalWriter":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()


def parse_timestamp(raw_value: object, *, default_timezone: str = "UTC") -> datetime | None:
    if raw_value is None:
        return None
    tzinfo = ZoneInfo(default_timezone)
    if isinstance(raw_value, datetime):
        return raw_value if raw_value.tzinfo else raw_value.replace(tzinfo=tzinfo)
    if isinstance(raw_value, (int, float)):
        return datetime.fromtimestamp(float(raw_value), tz=timezone.utc)

    text = str(raw_value).strip()
    if not text:
        return None

    try:
        parsed = parsedate_to_datetime(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=tzinfo)
    except Exception:
        pass

    for pattern in COMMON_TIME_PATTERNS:
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.replace(tzinfo=tzinfo)
        except ValueError:
            continue

    if text.endswith("Z"):
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=tzinfo)


def parse_chinese_month_day(
    raw_value: object,
    *,
    fallback_year: int,
    default_timezone: str = "UTC",
) -> datetime | None:
    if raw_value is None:
        return None
    match = CHINESE_MONTH_DAY_PATTERN.match(str(raw_value))
    if not match:
        return None
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour = int(match.group("hour") or 0)
    minute = int(match.group("minute") or 0)
    try:
        return datetime(fallback_year, month, day, hour, minute, tzinfo=ZoneInfo(default_timezone))
    except ValueError:
        return None


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def relative_seconds(source_time: datetime | None, event_time: datetime | None) -> float | None:
    if source_time is None or event_time is None:
        return None
    return (event_time - source_time).total_seconds()


def relative_hours(source_time: datetime | None, event_time: datetime | None) -> float | None:
    seconds = relative_seconds(source_time, event_time)
    return None if seconds is None else seconds / 3600.0


def coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.endswith("+"):
        text = text[:-1]
    try:
        return int(float(text))
    except ValueError:
        return None


def has_url(text: str | None) -> bool:
    if not text:
        return False
    return bool(URL_PATTERN.search(text))


def punctuation_intensity(text: str | None) -> float:
    if not text:
        return 0.0
    punct = len(PUNCTUATION_PATTERN.findall(text))
    return punct / max(len(text), 1)


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
