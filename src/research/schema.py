from __future__ import annotations


CANONICAL_FIELDS = [
    "dataset",
    "language",
    "event_id",
    "thread_id",
    "source_id",
    "post_id",
    "parent_id",
    "post_type",
    "interaction_type",
    "timestamp_raw",
    "timestamp_iso",
    "timestamp_relative",
    "timestamp_relative_hours",
    "text",
    "user_id",
    "label",
    "thread_label",
    "is_rumour",
    "veracity",
    "reaction_count",
    "repost_count",
    "like_count",
    "stance_type",
    "inferred_correction_signal",
    "publisher_type",
    "publisher_type_source",
    "explicit_correction",
    "inferred_correction",
    "is_correction",
    "correction_type",
    "metadata",
]


def normalize_record(record: dict) -> dict:
    normalized = {field: record.get(field) for field in CANONICAL_FIELDS}
    metadata = normalized.get("metadata")
    normalized["metadata"] = metadata if isinstance(metadata, dict) else {}
    return normalized
