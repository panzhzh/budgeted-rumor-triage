from __future__ import annotations

import csv
from pathlib import Path

from .config import DEFAULT_WINDOWS_HOURS
from .io_utils import (
    dump_json,
    has_url,
    punctuation_intensity,
)


def extract_thread_features(
    canonical_csv_path: Path,
    dataset_name: str,
    features_csv_path: Path,
    summary_path: Path,
    windows_hours: tuple[int, ...] = DEFAULT_WINDOWS_HOURS,
) -> dict:
    thread_states = {}
    summary = {
        "dataset": dataset_name,
        "windows_hours": list(windows_hours),
    }

    features_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with canonical_csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            state = thread_states.setdefault(row["thread_id"], _new_thread_state(row, windows_hours))
            if row["post_type"] == "source":
                _update_source_state(state, row)
            else:
                _update_reaction_state(state, row, windows_hours)

    fieldnames = _feature_fieldnames(windows_hours)
    feature_rows = []
    with features_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for thread_id in sorted(thread_states):
            feature_row = _finalize_feature_row(thread_states[thread_id], windows_hours)
            writer.writerow(feature_row)
            feature_rows.append(feature_row)

    summary["threads"] = len(thread_states)
    summary["feature_rows"] = len(feature_rows)
    dump_json(summary_path, summary)
    return summary


def _new_thread_state(row: dict, windows_hours: tuple[int, ...]) -> dict:
    return {
        "dataset": row.get("dataset"),
        "thread_id": row.get("thread_id"),
        "event_id": row.get("event_id"),
        "thread_label": row.get("thread_label"),
        "source_text": "",
        "source_stance_type": None,
        "source_publisher_type": None,
        "source_is_correction": 0,
        "correction_post_count": 0,
        "first_correction_hours": None,
        "first_official_correction_hours": None,
        "first_correction_type": None,
        "reaction_total": 0,
        "first_reaction_hours": None,
        "last_reaction_hours": None,
        "response_counts": {"deny": 0, "query": 0, "evidence": 0, "emotion": 0, "other": 0},
        "publisher_counts": {"official": 0, "expert": 0, "media": 0, "user": 0},
        "window_counts": {
            window: {
                "reactions": 0,
                "deny": 0,
                "query": 0,
                "evidence": 0,
                "emotion": 0,
                "other": 0,
                "corrections": 0,
                "official_corrections": 0,
            }
            for window in windows_hours
        },
    }


def _update_source_state(state: dict, row: dict) -> None:
    state["event_id"] = row.get("event_id") or state["event_id"]
    state["thread_label"] = row.get("thread_label") or state["thread_label"]
    state["source_text"] = row.get("text") or ""
    state["source_stance_type"] = row.get("stance_type")
    state["source_publisher_type"] = row.get("publisher_type")
    state["source_is_correction"] = int(str(row.get("is_correction") or "").lower() in {"1", "true", "yes"})


def _update_reaction_state(state: dict, row: dict, windows_hours: tuple[int, ...]) -> None:
    state["reaction_total"] += 1
    hours = _float_or_none(row.get("timestamp_relative_hours"))
    if hours is not None:
        if state["first_reaction_hours"] is None or hours < state["first_reaction_hours"]:
            state["first_reaction_hours"] = hours
        if state["last_reaction_hours"] is None or hours > state["last_reaction_hours"]:
            state["last_reaction_hours"] = hours
    label = row.get("stance_type") or "other"
    if label not in state["response_counts"]:
        label = "other"
    state["response_counts"][label] += 1
    publisher_type = row.get("publisher_type") or "user"
    if publisher_type not in state["publisher_counts"]:
        publisher_type = "user"
    state["publisher_counts"][publisher_type] += 1

    is_correction = str(row.get("is_correction") or "").lower() in {"1", "true", "yes"}
    if is_correction:
        state["correction_post_count"] += 1
        if state["first_correction_type"] is None:
            state["first_correction_type"] = row.get("correction_type")
        if hours is not None and (
            state["first_correction_hours"] is None or hours < state["first_correction_hours"]
        ):
            state["first_correction_hours"] = hours
        if (
            publisher_type in {"official", "expert"}
            and hours is not None
            and (
                state["first_official_correction_hours"] is None
                or hours < state["first_official_correction_hours"]
            )
        ):
            state["first_official_correction_hours"] = hours

    for window in windows_hours:
        if hours is not None and hours <= window:
            state["window_counts"][window]["reactions"] += 1
            state["window_counts"][window][label] += 1
            if is_correction:
                state["window_counts"][window]["corrections"] += 1
                if publisher_type in {"official", "expert"}:
                    state["window_counts"][window]["official_corrections"] += 1


def _finalize_feature_row(
    state: dict,
    windows_hours: tuple[int, ...],
) -> dict:
    feature_row = {
        "dataset": state["dataset"],
        "thread_id": state["thread_id"],
        "event_id": state["event_id"],
        "thread_label": state["thread_label"],
        "reaction_total": state["reaction_total"],
        "source_length": len(state["source_text"] or ""),
        "source_has_url": int(has_url(state["source_text"])),
        "source_punctuation_intensity": punctuation_intensity(state["source_text"]),
        "source_stance_type": state["source_stance_type"],
        "source_publisher_type": state["source_publisher_type"],
        "source_is_correction": state["source_is_correction"],
        "first_reaction_hours": state["first_reaction_hours"],
        "last_reaction_hours": state["last_reaction_hours"],
        "correction_post_count": state["correction_post_count"],
        "first_correction_hours": state["first_correction_hours"],
        "first_official_correction_hours": state["first_official_correction_hours"],
        "first_correction_type": state["first_correction_type"],
    }

    for window in windows_hours:
        window_counts = state["window_counts"][window]
        reactions = window_counts["reactions"]
        feature_row[f"window_{window}h_reactions"] = reactions
        feature_row[f"window_{window}h_growth_rate"] = round(reactions / max(window, 1), 4)
        feature_row[f"window_{window}h_corrections"] = window_counts["corrections"]
        feature_row[f"window_{window}h_official_corrections"] = window_counts["official_corrections"]
        for response_type in ("deny", "query", "evidence", "emotion", "other"):
            count = window_counts[response_type]
            feature_row[f"window_{window}h_{response_type}_count"] = count
            feature_row[f"window_{window}h_{response_type}_ratio"] = round(
                count / reactions, 4
            ) if reactions else 0.0

    feature_row["dominant_stance_type"] = _dominant_response(state["response_counts"])
    feature_row["query_or_unverified_ratio_6h"] = round(
        (
            state["response_counts"]["query"] +
            (state["reaction_total"] if state["thread_label"] in {"unclear", "unverified"} else 0)
        ) / max(state["reaction_total"], 1),
        4,
    )
    feature_row["evidence_ratio"] = round(
        state["response_counts"]["evidence"] / max(state["reaction_total"], 1),
        4,
    )
    for publisher_type, count in sorted(state["publisher_counts"].items()):
        feature_row[f"publisher_{publisher_type}_count"] = count
        feature_row[f"publisher_{publisher_type}_ratio"] = round(
            count / max(state["reaction_total"], 1),
            4,
        )
    return feature_row


def _feature_fieldnames(windows_hours: tuple[int, ...]) -> list[str]:
    fields = [
        "dataset",
        "thread_id",
        "event_id",
        "thread_label",
        "reaction_total",
        "source_length",
        "source_has_url",
        "source_punctuation_intensity",
        "source_stance_type",
        "source_publisher_type",
        "source_is_correction",
        "first_reaction_hours",
        "last_reaction_hours",
        "correction_post_count",
        "first_correction_hours",
        "first_official_correction_hours",
        "first_correction_type",
    ]
    for window in windows_hours:
        fields.extend(
            [
                f"window_{window}h_reactions",
                f"window_{window}h_growth_rate",
                f"window_{window}h_corrections",
                f"window_{window}h_official_corrections",
                f"window_{window}h_deny_count",
                f"window_{window}h_deny_ratio",
                f"window_{window}h_query_count",
                f"window_{window}h_query_ratio",
                f"window_{window}h_evidence_count",
                f"window_{window}h_evidence_ratio",
                f"window_{window}h_emotion_count",
                f"window_{window}h_emotion_ratio",
                f"window_{window}h_other_count",
                f"window_{window}h_other_ratio",
            ]
        )
    fields.extend(
        [
            "dominant_stance_type",
            "query_or_unverified_ratio_6h",
            "evidence_ratio",
            "publisher_expert_count",
            "publisher_expert_ratio",
            "publisher_media_count",
            "publisher_media_ratio",
            "publisher_official_count",
            "publisher_official_ratio",
            "publisher_user_count",
            "publisher_user_ratio",
        ]
    )
    return fields


def _dominant_response(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return max(sorted(counts), key=lambda key: counts[key])


def _float_or_none(value: object) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
