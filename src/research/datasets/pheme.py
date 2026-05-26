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


def parse_pheme(
    *,
    dataset_path: Path,
    jsonl_path: Path,
    csv_path: Path,
    audit_path: Path,
    default_timezone: str = "UTC",
) -> dict:
    root = dataset_path / "all-rnr-annotated-threads"
    audit = {
        "dataset": "PHEME",
        "dataset_path": str(root),
        "events": 0,
        "thread_count": 0,
        "source_posts": 0,
        "reaction_posts": 0,
        "timestamp_parse_failures": 0,
        "missing_text": 0,
        "missing_user": 0,
        "annotation_distribution": Counter(),
        "post_type_distribution": Counter(),
        "threads_without_reactions": 0,
        "correction_posts": 0,
    }

    with CanonicalWriter(jsonl_path, csv_path) as writer:
        for event_dir in sorted(root.iterdir()):
            if not event_dir.is_dir() or event_dir.name.startswith("."):
                continue
            audit["events"] += 1
            for label_dir in sorted(event_dir.iterdir()):
                if not label_dir.is_dir() or label_dir.name.startswith("."):
                    continue
                for thread_dir in sorted(label_dir.iterdir()):
                    if not thread_dir.is_dir() or thread_dir.name.startswith("."):
                        continue
                    audit["thread_count"] += 1
                    _write_pheme_thread(
                        writer=writer,
                        audit=audit,
                        event_dir=event_dir,
                        label_dir=label_dir,
                        thread_dir=thread_dir,
                        default_timezone=default_timezone,
                    )

    final_audit = dict(audit)
    final_audit["annotation_distribution"] = dict(audit["annotation_distribution"])
    final_audit["post_type_distribution"] = dict(audit["post_type_distribution"])
    dump_json(audit_path, final_audit)
    return final_audit


def _write_pheme_thread(
    *,
    writer: CanonicalWriter,
    audit: dict,
    event_dir: Path,
    label_dir: Path,
    thread_dir: Path,
    default_timezone: str,
) -> None:
    source_files = list(iter_json_files(thread_dir / "source-tweets"))
    if not source_files:
        audit["threads_without_reactions"] += 1
        return

    source_payload = load_json(source_files[0])
    annotation = load_json(thread_dir / "annotation.json") if (thread_dir / "annotation.json").exists() else {}
    structure = load_json(thread_dir / "structure.json") if (thread_dir / "structure.json").exists() else {}

    source_id = safe_text(source_payload.get("id_str")) or safe_text(source_payload.get("id")) or thread_dir.name
    source_time = parse_timestamp(source_payload.get("created_at"), default_timezone=default_timezone)
    if source_time is None:
        audit["timestamp_parse_failures"] += 1

    source_text = safe_text(source_payload.get("text"))
    if not source_text:
        audit["missing_text"] += 1
    source_user = safe_text(source_payload.get("user", {}).get("id_str")) or safe_text(source_payload.get("user", {}).get("id"))
    if not source_user:
        audit["missing_user"] += 1

    annotation_label = annotation.get("is_rumour", label_dir.name)
    audit["annotation_distribution"][annotation_label] += 1

    source_metadata = {
        "category": annotation.get("category"),
        "misinformation": annotation.get("misinformation"),
        "true": annotation.get("true"),
        "is_turnaround": annotation.get("is_turnaround"),
        "links": annotation.get("links"),
        "lang": source_payload.get("lang"),
        "structure_root_keys": list(structure.keys()) if isinstance(structure, dict) else None,
        "punctuation_intensity": punctuation_intensity(source_text),
        "user_verified": bool(source_payload.get("user", {}).get("verified")),
        "screen_name": source_payload.get("user", {}).get("screen_name"),
        "followers_count": source_payload.get("user", {}).get("followers_count"),
    }
    source_signal = extract_intervention_signals(
        source_text,
        "en",
        metadata=source_metadata,
        explicit_publisher_type="official" if source_metadata["user_verified"] else None,
    )
    writer.write(
        {
            "dataset": "PHEME",
            "language": "en",
            "event_id": event_dir.name,
            "thread_id": thread_dir.name,
            "source_id": source_id,
            "post_id": source_id,
            "parent_id": None,
            "post_type": "source",
            "interaction_type": "source",
            "timestamp_raw": source_payload.get("created_at"),
            "timestamp_iso": to_iso(source_time),
            "timestamp_relative": 0.0 if source_time else None,
            "timestamp_relative_hours": 0.0 if source_time else None,
            "text": source_text,
            "user_id": source_user or None,
            "label": label_dir.name,
            "thread_label": annotation_label,
            "is_rumour": annotation_label == "rumour",
            "veracity": _resolve_veracity(annotation),
            "reaction_count": len(list(iter_json_files(thread_dir / "reactions"))),
            "repost_count": coerce_int(source_payload.get("retweet_count")),
            "like_count": coerce_int(source_payload.get("favorite_count")),
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

    reaction_files = list(iter_json_files(thread_dir / "reactions"))
    if not reaction_files:
        audit["threads_without_reactions"] += 1
    for index, reaction_path in enumerate(reaction_files):
        reaction = load_json(reaction_path)
        event_time = parse_timestamp(reaction.get("created_at"), default_timezone=default_timezone)
        if event_time is None:
            audit["timestamp_parse_failures"] += 1
        text = safe_text(reaction.get("text"))
        if not text:
            audit["missing_text"] += 1
        reaction_user = safe_text(reaction.get("user", {}).get("id_str")) or safe_text(reaction.get("user", {}).get("id"))
        if not reaction_user:
            audit["missing_user"] += 1
        reaction_metadata = {
            "reaction_index": index,
            "lang": reaction.get("lang"),
            "user_verified": bool(reaction.get("user", {}).get("verified")),
            "screen_name": reaction.get("user", {}).get("screen_name"),
            "followers_count": reaction.get("user", {}).get("followers_count"),
        }
        reaction_signal = extract_intervention_signals(
            text,
            "en",
            metadata=reaction_metadata,
            explicit_publisher_type="official" if reaction_metadata["user_verified"] else None,
        )
        post_id = safe_text(reaction.get("id_str")) or safe_text(reaction.get("id")) or reaction_path.stem
        parent_id = safe_text(reaction.get("in_reply_to_status_id_str")) or safe_text(reaction.get("in_reply_to_status_id")) or source_id
        writer.write(
            {
                "dataset": "PHEME",
                "language": "en",
                "event_id": event_dir.name,
                "thread_id": thread_dir.name,
                "source_id": source_id,
                "post_id": post_id,
                "parent_id": parent_id,
                "post_type": "reaction",
                "interaction_type": "reply",
                "timestamp_raw": reaction.get("created_at"),
                "timestamp_iso": to_iso(event_time),
                "timestamp_relative": relative_seconds(source_time, event_time),
                "timestamp_relative_hours": relative_hours(source_time, event_time),
                "text": text,
                "user_id": reaction_user or None,
                "label": "reply",
                "thread_label": annotation_label,
                "is_rumour": annotation_label == "rumour",
                "veracity": _resolve_veracity(annotation),
                "reaction_count": None,
                "repost_count": coerce_int(reaction.get("retweet_count")),
                "like_count": coerce_int(reaction.get("favorite_count")),
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
                    **reaction_metadata,
                },
            }
        )
        audit["reaction_posts"] += 1
        audit["post_type_distribution"]["reaction"] += 1


def _resolve_veracity(annotation: dict) -> str | None:
    if not annotation:
        return None
    if annotation.get("true") == 1:
        return "true"
    if annotation.get("misinformation") == 1:
        return "false"
    if annotation.get("true") == 0 and annotation.get("misinformation") == 0:
        return "unverified"
    return None
