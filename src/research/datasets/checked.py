from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..io_utils import (
    CanonicalWriter,
    coerce_int,
    dump_json,
    iter_json_files,
    load_json,
    parse_timestamp,
    punctuation_intensity,
    relative_hours,
    relative_seconds,
    safe_text,
    to_iso,
)
from ..text_rules import extract_intervention_signals


def _init_audit(dataset_name: str, dataset_path: Path) -> dict:
    return {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "source_posts": 0,
        "reaction_posts": 0,
        "thread_count": 0,
        "files_seen": 0,
        "files_skipped": [],
        "timestamp_parse_failures": 0,
        "missing_text": 0,
        "missing_user": 0,
        "correction_posts": 0,
        "label_distribution": Counter(),
        "post_type_distribution": Counter(),
    }


def parse_checked(
    *,
    dataset_path: Path,
    jsonl_path: Path,
    csv_path: Path,
    audit_path: Path,
    default_timezone: str = "Asia/Shanghai",
) -> dict:
    audit = _init_audit("CHECKED", dataset_path)
    label_to_dir = {
        "fake": dataset_path / "dataset" / "fake_news",
        "real": dataset_path / "dataset" / "real_news",
    }

    with CanonicalWriter(jsonl_path, csv_path) as writer:
        for thread_label, directory in label_to_dir.items():
            json_files = list(iter_json_files(directory))
            audit["thread_count"] += len(json_files)
            for path in json_files:
                audit["files_seen"] += 1
                try:
                    payload = load_json(path)
                except Exception as exc:
                    audit["files_skipped"].append({"file": str(path), "reason": str(exc)})
                    continue
                audit["label_distribution"][thread_label] += 1
                _write_checked_thread(
                    writer=writer,
                    audit=audit,
                    payload=payload,
                    thread_label=thread_label,
                    source_filename=path.name,
                    default_timezone=default_timezone,
                )

    final_audit = dict(audit)
    final_audit["label_distribution"] = dict(audit["label_distribution"])
    final_audit["post_type_distribution"] = dict(audit["post_type_distribution"])
    dump_json(audit_path, final_audit)
    return final_audit


def _write_checked_thread(
    *,
    writer: CanonicalWriter,
    audit: dict,
    payload: dict,
    thread_label: str,
    source_filename: str,
    default_timezone: str,
) -> None:
    source_id = safe_text(payload.get("id")) or source_filename.rsplit(".", 1)[0]
    source_time = parse_timestamp(payload.get("date"), default_timezone=default_timezone)
    if source_time is None:
        audit["timestamp_parse_failures"] += 1
    source_text = safe_text(payload.get("text"))
    if not source_text:
        audit["missing_text"] += 1
    user_id = safe_text(payload.get("user_id"))
    if not user_id:
        audit["missing_user"] += 1

    thread_id = source_id
    source_metadata = {
        "analysis": payload.get("analysis"),
        "pic_url_count": len(payload.get("pic_url", []) or []),
        "video_url": payload.get("video_url"),
        "punctuation_intensity": punctuation_intensity(source_text),
        "fact_check_available": bool(payload.get("analysis")),
    }
    source_signal = extract_intervention_signals(
        source_text,
        "zh",
        metadata=source_metadata,
        explicit_publisher_type="user",
        explicit_is_correction=False,
    )
    writer.write(
        {
            "dataset": "CHECKED",
            "language": "zh",
            "event_id": thread_id,
            "thread_id": thread_id,
            "source_id": source_id,
            "post_id": source_id,
            "parent_id": None,
            "post_type": "source",
            "interaction_type": "source",
            "timestamp_raw": payload.get("date"),
            "timestamp_iso": to_iso(source_time),
            "timestamp_relative": 0.0 if source_time else None,
            "timestamp_relative_hours": 0.0 if source_time else None,
            "text": source_text,
            "user_id": user_id or None,
            "label": payload.get("label"),
            "thread_label": thread_label,
            "is_rumour": thread_label == "fake",
            "veracity": payload.get("label"),
            "reaction_count": coerce_int(payload.get("comment_num")),
            "repost_count": coerce_int(payload.get("repost_num")),
            "like_count": coerce_int(payload.get("like_num")),
            "stance_type": source_signal.stance_type,
            "inferred_correction_signal": source_signal.inferred_correction_signal,
            "publisher_type": source_signal.publisher_type,
            "publisher_type_source": source_signal.publisher_type_source,
            "explicit_correction": source_signal.explicit_correction,
            "inferred_correction": source_signal.inferred_correction,
            "is_correction": source_signal.is_correction,
            "correction_type": source_signal.correction_type,
            "metadata": source_metadata,
        }
    )
    audit["source_posts"] += 1
    audit["post_type_distribution"]["source"] += 1

    analysis_text = safe_text(payload.get("analysis"))
    if analysis_text:
        correction_signal = extract_intervention_signals(
            analysis_text,
            "zh",
            metadata={"fact_check_available": True},
            explicit_publisher_type="official",
            explicit_is_correction=True,
            explicit_correction_type="fact_check",
        )
        writer.write(
            {
                "dataset": "CHECKED",
                "language": "zh",
                "event_id": thread_id,
                "thread_id": thread_id,
                "source_id": source_id,
                "post_id": f"{source_id}:fact_check",
                "parent_id": source_id,
                "post_type": "correction",
                "interaction_type": "fact_check",
                "timestamp_raw": None,
                "timestamp_iso": None,
                "timestamp_relative": None,
                "timestamp_relative_hours": None,
                "text": analysis_text,
                "user_id": "weibo_fact_check",
                "label": "fact_check",
                "thread_label": thread_label,
                "is_rumour": thread_label == "fake",
                "veracity": payload.get("label"),
                "reaction_count": None,
                "repost_count": None,
                "like_count": None,
                "stance_type": correction_signal.stance_type,
                "inferred_correction_signal": correction_signal.inferred_correction_signal,
                "publisher_type": correction_signal.publisher_type,
                "publisher_type_source": correction_signal.publisher_type_source,
                "explicit_correction": correction_signal.explicit_correction,
                "inferred_correction": correction_signal.inferred_correction,
                "is_correction": correction_signal.is_correction,
                "correction_type": correction_signal.correction_type,
                "metadata": {
                    "timestamp_missing": True,
                    "analysis_source": True,
                    "fact_check_available": True,
                },
            }
        )
        audit["correction_posts"] += 1
        audit["post_type_distribution"]["correction"] += 1

    reaction_counter = 0
    for interaction_type, key in (("comment", "comments"), ("repost", "reposts")):
        items = payload.get(key) or []
        for index, item in enumerate(items):
            post_id = safe_text(item.get("id")) or f"{source_id}:{interaction_type}:{index}"
            event_time = parse_timestamp(item.get("date"), default_timezone=default_timezone)
            if event_time is None:
                audit["timestamp_parse_failures"] += 1
            text = safe_text(item.get("text"))
            if not text:
                audit["missing_text"] += 1
            reaction_user = safe_text(item.get("user_id"))
            if not reaction_user:
                audit["missing_user"] += 1
            reaction_signal = extract_intervention_signals(text, "zh")
            writer.write(
                {
                    "dataset": "CHECKED",
                    "language": "zh",
                    "event_id": thread_id,
                    "thread_id": thread_id,
                    "source_id": source_id,
                    "post_id": post_id,
                    "parent_id": source_id,
                    "post_type": "reaction",
                    "interaction_type": interaction_type,
                    "timestamp_raw": item.get("date"),
                    "timestamp_iso": to_iso(event_time),
                    "timestamp_relative": relative_seconds(source_time, event_time),
                    "timestamp_relative_hours": relative_hours(source_time, event_time),
                    "text": text,
                    "user_id": reaction_user or None,
                    "label": interaction_type,
                    "thread_label": thread_label,
                    "is_rumour": thread_label == "fake",
                    "veracity": payload.get("label"),
                    "reaction_count": None,
                    "repost_count": None,
                    "like_count": None,
                    "stance_type": reaction_signal.stance_type,
                    "inferred_correction_signal": reaction_signal.inferred_correction_signal,
                    "publisher_type": reaction_signal.publisher_type,
                    "publisher_type_source": reaction_signal.publisher_type_source,
                    "explicit_correction": reaction_signal.explicit_correction,
                    "inferred_correction": reaction_signal.inferred_correction,
                    "is_correction": reaction_signal.is_correction,
                    "correction_type": reaction_signal.correction_type,
                    "metadata": {
                        "rule_scores": reaction_signal.scores,
                        "pic_url_count": len(item.get("pic_url", []) or []),
                        "reaction_index": reaction_counter,
                    },
                }
            )
            reaction_counter += 1
            audit["reaction_posts"] += 1
            audit["post_type_distribution"]["reaction"] += 1
