from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from .config import DATASET_NAMES
from .io_utils import dump_json, load_json


def dataset_quality_report(canonical_csv_path: Path, dataset_name: str) -> dict:
    total_records = 0
    source_posts = 0
    reaction_posts = 0
    thread_ids = set()
    event_ids = set()
    timestamp_present = 0
    missing_text = 0
    missing_user = 0
    response_distribution = {}
    label_distribution = {}
    per_thread_reactions = {}
    relative_values = []

    with canonical_csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_records += 1
            thread_ids.add(row["thread_id"])
            event_ids.add(row["event_id"])
            if row.get("post_type") == "source":
                source_posts += 1
            if row.get("post_type") == "reaction":
                reaction_posts += 1
                per_thread_reactions[row["thread_id"]] = per_thread_reactions.get(row["thread_id"], 0) + 1
                if row.get("timestamp_relative_hours") not in {"", None}:
                    relative_values.append(float(row["timestamp_relative_hours"]))
            else:
                per_thread_reactions.setdefault(row["thread_id"], 0)

            if row.get("timestamp_iso"):
                timestamp_present += 1
            if not row.get("text"):
                missing_text += 1
            if not row.get("user_id"):
                missing_user += 1
            response_key = row.get("stance_type") or "missing"
            label_key = row.get("thread_label") or "missing"
            response_distribution[response_key] = response_distribution.get(response_key, 0) + 1
            label_distribution[label_key] = label_distribution.get(label_key, 0) + 1

    return {
        "dataset": dataset_name,
        "total_records": total_records,
        "source_posts": source_posts,
        "reaction_posts": reaction_posts,
        "thread_count": len(thread_ids),
        "event_count": len(event_ids),
        "timestamp_coverage_rate": _rate(timestamp_present, total_records),
        "missing_text_rate": _rate(missing_text, total_records),
        "missing_user_rate": _rate(missing_user, total_records),
        "stance_type_distribution": response_distribution,
        "thread_label_distribution": label_distribution,
        "avg_reactions_per_thread": _avg_by_thread(per_thread_reactions),
        "avg_relative_hours": round(mean(relative_values), 4) if relative_values else None,
        "max_relative_hours": round(max(relative_values), 4) if relative_values else None,
    }


def cross_dataset_report(report_paths: dict[str, Path], output_path: Path) -> dict:
    datasets = {}
    for name in DATASET_NAMES:
        path = report_paths.get(name)
        if path and path.exists():
            datasets[name] = load_json(path)
    cross = {
        "datasets": datasets,
        "comparison": {
            name: {
                "thread_count": payload.get("thread_count"),
                "reaction_posts": payload.get("reaction_posts"),
                "timestamp_coverage_rate": payload.get("timestamp_coverage_rate"),
                "stance_types": sorted((payload.get("stance_type_distribution") or {}).keys()),
            }
            for name, payload in datasets.items()
        },
    }
    dump_json(output_path, cross)
    return cross


def _avg_by_thread(per_thread_reactions: dict[str, int]) -> float | None:
    if not per_thread_reactions:
        return None
    return round(sum(per_thread_reactions.values()) / len(per_thread_reactions), 4)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)
