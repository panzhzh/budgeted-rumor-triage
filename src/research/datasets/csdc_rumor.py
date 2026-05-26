from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..io_utils import (
    CanonicalWriter,
    dump_json,
    load_json,
    load_json_lines,
    parse_chinese_month_day,
    parse_timestamp,
    punctuation_intensity,
    relative_hours,
    relative_seconds,
    safe_text,
    to_iso,
)
from ..text_rules import extract_intervention_signals


def parse_csdc_rumor(
    *,
    dataset_path: Path,
    jsonl_path: Path,
    csv_path: Path,
    audit_path: Path,
    default_timezone: str = "Asia/Shanghai",
) -> dict:
    audit = {
        "dataset": "CSDC-Rumor",
        "dataset_path": str(dataset_path),
        "source_posts": 0,
        "reaction_posts": 0,
        "thread_count": 0,
        "timestamp_parse_failures": 0,
        "missing_text": 0,
        "missing_user": 0,
        "missing_fact_match": 0,
        "correction_posts": 0,
        "post_type_distribution": Counter(),
    }

    facts = load_json_lines(dataset_path / "fact.json")
    facts_by_date = {}
    for row in facts:
        facts_by_date.setdefault(row.get("date"), []).append(row)

    rumor_dir = dataset_path / "rumor_weibo"
    interaction_dir = dataset_path / "rumor_forward_comment"

    with CanonicalWriter(jsonl_path, csv_path) as writer:
        for rumor_path in sorted(rumor_dir.glob("*.json")):
            if rumor_path.name.startswith("._") or rumor_path.name == ".DS_Store":
                continue
            audit["thread_count"] += 1
            payload = load_json(rumor_path)
            interaction_path = interaction_dir / rumor_path.name
            interactions = load_json(interaction_path) if interaction_path.exists() else []
            fact_candidates = facts_by_date.get(payload.get("publishTime"), [])
            fact_entry = fact_candidates[0] if fact_candidates else None
            if fact_entry is None:
                audit["missing_fact_match"] += 1
            _write_csdc_thread(
                writer=writer,
                audit=audit,
                payload=payload,
                interactions=interactions,
                fact_entry=fact_entry,
                default_timezone=default_timezone,
            )

    final_audit = dict(audit)
    final_audit["post_type_distribution"] = dict(audit["post_type_distribution"])
    dump_json(audit_path, final_audit)
    return final_audit


def _write_csdc_thread(
    *,
    writer: CanonicalWriter,
    audit: dict,
    payload: dict,
    interactions: list,
    fact_entry: dict | None,
    default_timezone: str,
) -> None:
    source_id = safe_text(payload.get("rumorCode")) or safe_text(payload.get("title"))
    thread_id = source_id
    source_time = parse_timestamp(payload.get("publishTime"), default_timezone=default_timezone)
    if source_time is None:
        audit["timestamp_parse_failures"] += 1
    text = safe_text(payload.get("rumorText")) or safe_text(payload.get("title"))
    if not text:
        audit["missing_text"] += 1

    source_user = safe_text(payload.get("rumormongerName")) or safe_text(payload.get("informerName"))
    if not source_user:
        audit["missing_user"] += 1

    source_metadata = {
        "title": payload.get("title"),
        "informer_name": payload.get("informerName"),
        "visit_times": payload.get("visitTimes"),
        "related_url_count": len(payload.get("related_url", []) or []),
        "fact_title": fact_entry.get("title") if fact_entry else None,
        "fact_explain": fact_entry.get("explain") if fact_entry else None,
        "fact_tags": fact_entry.get("tag") if fact_entry else None,
        "fact_date": fact_entry.get("date") if fact_entry else None,
        "fact_check_available": fact_entry is not None,
        "punctuation_intensity": punctuation_intensity(text),
    }
    source_signal = extract_intervention_signals(
        text,
        "zh",
        metadata=source_metadata,
        explicit_publisher_type="user",
        explicit_is_correction=False,
    )
    writer.write(
        {
            "dataset": "CSDC-Rumor",
            "language": "zh",
            "event_id": thread_id,
            "thread_id": thread_id,
            "source_id": source_id,
            "post_id": source_id,
            "parent_id": None,
            "post_type": "source",
            "interaction_type": "source",
            "timestamp_raw": payload.get("publishTime"),
            "timestamp_iso": to_iso(source_time),
            "timestamp_relative": 0.0 if source_time else None,
            "timestamp_relative_hours": 0.0 if source_time else None,
            "text": text,
            "user_id": source_user or None,
            "label": payload.get("result"),
            "thread_label": "rumor",
            "is_rumour": True,
            "veracity": payload.get("result"),
            "reaction_count": len(interactions),
            "repost_count": sum(1 for item in interactions if item.get("comment_or_forward") == "forward"),
            "like_count": None,
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

    if fact_entry:
        fact_time = parse_timestamp(fact_entry.get("date"), default_timezone=default_timezone)
        if fact_time is None:
            audit["timestamp_parse_failures"] += 1
        fact_text = "\n".join(
            part
            for part in (
                safe_text(fact_entry.get("title")),
                safe_text(fact_entry.get("abstract")),
                safe_text(fact_entry.get("explain")),
            )
            if part
        )
        fact_signal = extract_intervention_signals(
            fact_text,
            "zh",
            metadata={"fact_check_available": True},
            explicit_publisher_type="expert",
            explicit_is_correction=True,
            explicit_correction_type="fact_check",
        )
        writer.write(
            {
                "dataset": "CSDC-Rumor",
                "language": "zh",
                "event_id": thread_id,
                "thread_id": thread_id,
                "source_id": source_id,
                "post_id": f"{source_id}:fact_check",
                "parent_id": source_id,
                "post_type": "correction",
                "interaction_type": "fact_check",
                "timestamp_raw": fact_entry.get("date"),
                "timestamp_iso": to_iso(fact_time),
                "timestamp_relative": relative_seconds(source_time, fact_time),
                "timestamp_relative_hours": relative_hours(source_time, fact_time),
                "text": fact_text,
                "user_id": "fact_check_platform",
                "label": fact_entry.get("explain"),
                "thread_label": "rumor",
                "is_rumour": True,
                "veracity": payload.get("result"),
                "reaction_count": None,
                "repost_count": None,
                "like_count": None,
                "stance_type": fact_signal.stance_type,
                "inferred_correction_signal": fact_signal.inferred_correction_signal,
                "publisher_type": fact_signal.publisher_type,
                "publisher_type_source": fact_signal.publisher_type_source,
                "explicit_correction": fact_signal.explicit_correction,
                "inferred_correction": fact_signal.inferred_correction,
                "is_correction": fact_signal.is_correction,
                "correction_type": fact_signal.correction_type,
                "metadata": {
                    "fact_title": fact_entry.get("title"),
                    "fact_tags": fact_entry.get("tag"),
                    "fact_date": fact_entry.get("date"),
                    "fact_check_available": True,
                },
            }
        )
        audit["correction_posts"] += 1
        audit["post_type_distribution"]["correction"] += 1

    for index, item in enumerate(interactions):
        post_id = f"{source_id}:{item.get('comment_or_forward', 'reaction')}:{index}"
        event_time = parse_timestamp(item.get("date"), default_timezone=default_timezone)
        if event_time is None and source_time is not None:
            event_time = parse_chinese_month_day(
                item.get("date"),
                fallback_year=source_time.year,
                default_timezone=default_timezone,
            )
        if event_time is None:
            audit["timestamp_parse_failures"] += 1
        reaction_text = safe_text(item.get("text"))
        if not reaction_text:
            audit["missing_text"] += 1
        reaction_user = safe_text(item.get("uid"))
        if not reaction_user:
            audit["missing_user"] += 1
        reaction_signal = extract_intervention_signals(reaction_text, "zh")
        interaction_type = safe_text(item.get("comment_or_forward")) or "reaction"
        writer.write(
            {
                "dataset": "CSDC-Rumor",
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
                "text": reaction_text,
                "user_id": reaction_user or None,
                "label": interaction_type,
                "thread_label": "rumor",
                "is_rumour": True,
                "veracity": payload.get("result"),
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
                    "reaction_index": index,
                },
            }
        )
        audit["reaction_posts"] += 1
        audit["post_type_distribution"]["reaction"] += 1
