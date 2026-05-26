from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .feature_store import config_fingerprint, merge_unique_rows, read_parquet_frame, text_hash, write_frame
from .io_utils import dump_json, ensure_dir


PROTOTYPE_TEXTS = {
    "denial": "This is a direct denial or debunking message that explicitly rejects a rumor.",
    "evidence_explanation": "This is a corrective message that provides evidence, facts, links, or explanation.",
    "action_guidance": "This is a corrective message that gives advice, guidance, or recommended actions.",
    "authority_statement": "This is a corrective message from an official authority or expert institution.",
    "emotion_reassurance": "This is a corrective message that reassures people and reduces fear or panic.",
}

ZERO_SHOT_LABELS = (
    "denial",
    "evidence_explanation",
    "action_guidance",
    "authority_statement",
    "query",
)


def build_text_model_features(
    canonical_csv_path: Path,
    dataset_name: str,
    output_dir: Path,
    *,
    feature_store_dir: Path,
    sentence_model_name: str,
    transformer_model_name: str,
    batch_size: int = 32,
    text_model_level: str = "light",
    reuse_cache: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(feature_store_dir)

    rows = list(_iter_canonical_rows(canonical_csv_path))
    text_rows = _select_text_rows(rows)
    config_payload = {
        "dataset": dataset_name,
        "sentence_model_name": sentence_model_name,
        "transformer_model_name": transformer_model_name,
        "batch_size": batch_size,
        "text_model_level": text_model_level,
        "prototype_texts": PROTOTYPE_TEXTS,
        "zero_shot_labels": list(ZERO_SHOT_LABELS),
        "selection_rule": "source_posts_plus_early_correction_candidates_with_timestamp<=6h",
    }
    fingerprint = config_fingerprint(config_payload)

    post_feature_store = feature_store_dir / "post_semantic_features.parquet"
    post_feature_csv = output_dir / "post_semantic_features.csv"
    thread_feature_store = feature_store_dir / "thread_text_features.parquet"
    thread_feature_csv = output_dir / "thread_text_features.csv"
    manifest_path = feature_store_dir / "feature_manifest.json"

    existing_post_features = read_parquet_frame(post_feature_store) if reuse_cache else pd.DataFrame()
    existing_post_features = _filter_matching_cache(existing_post_features, fingerprint)

    source_rows = [row for row in text_rows if row.get("post_type") == "source"]
    correction_rows = [row for row in text_rows if _is_early_correction_candidate(row)]

    source_semantic = _encode_with_prototypes(
        rows=source_rows,
        dataset_name=dataset_name,
        model_name=sentence_model_name,
        batch_size=batch_size,
        config_id=fingerprint,
        existing_cache=existing_post_features,
    )
    correction_semantic = _encode_with_prototypes(
        rows=correction_rows,
        dataset_name=dataset_name,
        model_name=sentence_model_name,
        batch_size=batch_size,
        config_id=fingerprint,
        existing_cache=existing_post_features,
    )

    semantic_frames = [frame for frame in (source_semantic["frame"], correction_semantic["frame"]) if not frame.empty]
    post_features = pd.concat(semantic_frames, ignore_index=True) if semantic_frames else pd.DataFrame()

    zero_shot_summary: dict[str, Any] = {
        "status": "skipped",
        "reason": "text_model_level_light",
        "rows_scored": 0,
        "model_name": transformer_model_name,
    }
    zero_shot_frame = pd.DataFrame()
    if text_model_level == "heavy":
        zero_shot_result = _run_zero_shot_scores(
            rows=correction_rows,
            dataset_name=dataset_name,
            model_name=transformer_model_name,
            batch_size=batch_size,
            config_id=fingerprint,
            existing_cache=existing_post_features,
        )
        zero_shot_summary = zero_shot_result["summary"]
        zero_shot_frame = zero_shot_result["frame"]
        if not zero_shot_frame.empty:
            if post_features.empty:
                post_features = zero_shot_frame
            else:
                post_features = _merge_zero_shot_features(post_features, zero_shot_frame)

    if post_features.empty:
        post_features = pd.DataFrame(columns=_post_feature_columns())
    else:
        post_features = _ensure_post_feature_columns(post_features)

    merged_post_store = merge_unique_rows(existing_post_features, post_features, key="cache_key")
    write_frame(merged_post_store, parquet_path=post_feature_store)
    post_features.to_csv(post_feature_csv, index=False)

    thread_features = _aggregate_thread_text_features(
        rows=rows,
        dataset_name=dataset_name,
        post_feature_frame=post_features,
        include_zero_shot=text_model_level == "heavy",
    )
    write_frame(thread_features, parquet_path=thread_feature_store, csv_path=thread_feature_csv)

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "config_id": fingerprint,
        "selection_rule": "source_posts_plus_early_correction_candidates_with_timestamp<=6h",
        "sentence_transformer": source_semantic["summary"],
        "correction_sentence_transformer": correction_semantic["summary"],
        "zero_shot": zero_shot_summary,
        "thread_features": {
            "status": "completed" if not thread_features.empty else "skipped",
            "rows": int(len(thread_features)),
            "output_csv": str(thread_feature_csv),
            "output_parquet": str(thread_feature_store),
        },
        "post_features": {
            "rows": int(len(post_features)),
            "output_csv": str(post_feature_csv),
            "output_parquet": str(post_feature_store),
        },
        "text_model_level": text_model_level,
    }
    dump_json(output_dir / "text_model_summary.json", summary)
    dump_json(
        manifest_path,
        {
            **config_payload,
            "config_id": fingerprint,
            "post_feature_store": str(post_feature_store),
            "thread_feature_store": str(thread_feature_store),
            "post_rows": int(len(post_features)),
            "thread_rows": int(len(thread_features)),
        },
    )
    return summary


def _select_text_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if row.get("post_type") == "source" or _is_early_correction_candidate(row):
            selected.append(row)
    return selected


def _filter_matching_cache(frame: pd.DataFrame, config_id: str) -> pd.DataFrame:
    if frame.empty or "config_id" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["config_id"] == config_id].copy()


def _encode_with_prototypes(
    *,
    rows: list[dict[str, Any]],
    dataset_name: str,
    model_name: str,
    batch_size: int,
    config_id: str,
    existing_cache: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "status": "skipped",
        "rows_encoded": 0,
        "rows_reused": 0,
        "model_name": model_name,
        "device": "cpu",
    }
    if not rows:
        return {"frame": pd.DataFrame(columns=_post_feature_columns()), "summary": summary}

    existing_by_hash = {}
    if not existing_cache.empty and "text_hash" in existing_cache.columns:
        for _, cached_row in existing_cache.iterrows():
            existing_by_hash[str(cached_row["text_hash"])] = cached_row.to_dict()

    texts_to_encode = []
    row_payloads = []
    reused_rows: list[dict[str, Any]] = []
    for row in rows:
        hashed = text_hash(row.get("text"))
        cached = existing_by_hash.get(hashed)
        if cached:
            reused_rows.append(_project_cached_row(row, cached))
            continue
        texts_to_encode.append(row.get("text") or "")
        row_payloads.append((row, hashed))

    encoded_rows: list[dict[str, Any]] = []
    device = "cpu"
    if texts_to_encode:
        from sentence_transformers import SentenceTransformer

        device = _resolve_text_model_device()
        model = SentenceTransformer(model_name, device=device)
        text_embeddings = model.encode(
            texts_to_encode,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        prototype_labels = list(PROTOTYPE_TEXTS)
        prototype_embeddings = model.encode(
            [PROTOTYPE_TEXTS[label] for label in prototype_labels],
            batch_size=min(batch_size, len(prototype_labels)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        similarities = text_embeddings @ prototype_embeddings.T
        topic_clusters, topic_distances = _fit_topic_clusters(text_embeddings)
        for (row, hashed), embedding, sim_vector, cluster_id, distance_to_center in zip(
            row_payloads,
            text_embeddings,
            similarities,
            topic_clusters,
            topic_distances,
        ):
            scores = {
                f"sim_{label}": round(float(score), 6)
                for label, score in zip(prototype_labels, sim_vector.tolist())
            }
            encoded_rows.append(
                _base_post_feature_row(
                    row,
                    hashed,
                    config_id,
                    {
                        "embedding_mean": round(float(np.mean(embedding)), 6),
                        "embedding_std": round(float(np.std(embedding)), 6),
                        "topic_cluster": int(cluster_id),
                        "topic_distance_to_center": round(float(distance_to_center), 6),
                        **scores,
                    },
                )
            )
        summary["status"] = "completed"
        summary["rows_encoded"] = len(encoded_rows)
    else:
        summary["status"] = "completed"
    summary["rows_reused"] = len(reused_rows)
    summary["device"] = device

    frame = pd.DataFrame(reused_rows + encoded_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=_post_feature_columns())
    else:
        frame = _ensure_post_feature_columns(frame)
    return {"frame": frame, "summary": summary}


def _run_zero_shot_scores(
    *,
    rows: list[dict[str, Any]],
    dataset_name: str,
    model_name: str,
    batch_size: int,
    config_id: str,
    existing_cache: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "status": "skipped",
        "rows_scored": 0,
        "rows_reused": 0,
        "model_name": model_name,
        "device": "cpu",
    }
    if not rows:
        return {"frame": pd.DataFrame(columns=["dataset", "post_id", "text_hash", "config_id"]), "summary": summary}

    zero_shot_cols = [f"zs_{label}" for label in ZERO_SHOT_LABELS]
    cached_zero_shot = existing_cache.copy()
    if not cached_zero_shot.empty:
        cached_zero_shot = cached_zero_shot.dropna(subset=["text_hash"])
        missing_cols = [column for column in zero_shot_cols if column not in cached_zero_shot.columns]
        if missing_cols:
            cached_zero_shot = pd.DataFrame()
        else:
            cached_zero_shot = cached_zero_shot.dropna(subset=zero_shot_cols, how="any")

    existing_by_hash = {}
    if not cached_zero_shot.empty:
        for _, cached_row in cached_zero_shot.iterrows():
            existing_by_hash[str(cached_row["text_hash"])] = cached_row.to_dict()

    rows_to_score = []
    reused_rows = []
    for row in rows:
        hashed = text_hash(row.get("text"))
        cached = existing_by_hash.get(hashed)
        if cached:
            payload = {
                "dataset": row.get("dataset"),
                "post_id": row.get("post_id"),
                "text_hash": hashed,
                "config_id": config_id,
                **{column: cached.get(column) for column in zero_shot_cols},
            }
            reused_rows.append(payload)
        else:
            rows_to_score.append((row, hashed))

    scored_rows: list[dict[str, Any]] = []
    device = "cpu"
    if rows_to_score:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if _torch_cuda_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device == "cuda":
            model = model.to(device)
            # Use fp16 by default for broader operator compatibility than bf16.
            model = model.to(dtype=torch.float16)
        model.eval()

        dataset = _ZeroShotPairDataset(rows_to_score, ZERO_SHOT_LABELS)
        loader = DataLoader(
            dataset,
            batch_size=max(batch_size, 1),
            shuffle=False,
            collate_fn=lambda items: _collate_zero_shot_batch(items, tokenizer),
            pin_memory=device == "cuda",
        )
        entailment_index = _resolve_entailment_index(model)
        contradiction_index = _resolve_contradiction_index(model)

        for batch in loader:
            batch_rows = batch["rows"]
            encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(**encoded).logits
            logits = logits.float()
            if contradiction_index is not None:
                selected_logits = logits[:, [contradiction_index, entailment_index]]
                entailment_scores = F.softmax(selected_logits, dim=-1)[:, 1]
            else:
                entailment_scores = F.softmax(logits, dim=-1)[:, entailment_index]
            entailment_scores = (
                entailment_scores.float().detach().cpu().numpy().reshape(-1, len(ZERO_SHOT_LABELS))
            )
            for row_info, score_vector in zip(batch_rows, entailment_scores):
                scored_rows.append(
                    {
                        "dataset": row_info["dataset"],
                        "post_id": row_info["post_id"],
                        "text_hash": row_info["text_hash"],
                        "config_id": config_id,
                        **{
                            f"zs_{label}": round(float(score), 6)
                            for label, score in zip(ZERO_SHOT_LABELS, score_vector.tolist())
                        },
                    }
                )

    frame = pd.DataFrame(reused_rows + scored_rows)
    summary.update(
        {
            "status": "completed",
            "rows_scored": int(len(scored_rows)),
            "rows_reused": int(len(reused_rows)),
            "rows_available": int(len(reused_rows) + len(scored_rows)),
            "device": device,
        }
    )
    return {"frame": frame, "summary": summary}


def _merge_zero_shot_features(post_features: pd.DataFrame, zero_shot_frame: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["dataset", "post_id", "text_hash", "config_id"]
    zero_shot_cols = [f"zs_{label}" for label in ZERO_SHOT_LABELS]
    update_cols = merge_keys + [column for column in zero_shot_cols if column in zero_shot_frame.columns]
    zero_shot_updates = zero_shot_frame[update_cols].drop_duplicates(subset=merge_keys, keep="last")
    merged = post_features.merge(
        zero_shot_updates,
        on=merge_keys,
        how="left",
        suffixes=("", "__zero_shot_update"),
    )
    for column in zero_shot_cols:
        update_column = f"{column}__zero_shot_update"
        if update_column in merged.columns:
            merged[column] = merged[update_column].combine_first(merged[column])
            merged = merged.drop(columns=[update_column])
        elif column not in merged.columns:
            merged[column] = np.nan
    return merged


def _aggregate_thread_text_features(
    *,
    rows: list[dict[str, Any]],
    dataset_name: str,
    post_feature_frame: pd.DataFrame,
    include_zero_shot: bool,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    feature_lookup = {}
    if not post_feature_frame.empty:
        for _, row in post_feature_frame.iterrows():
            feature_lookup[str(row["post_id"])] = row.to_dict()

    thread_state: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("post_type") != "source" and not _is_early_correction_candidate(row):
            continue
        thread_id = row["thread_id"]
        state = thread_state.setdefault(
            thread_id,
            {
                "dataset": dataset_name,
                "thread_id": thread_id,
                "source_text_count": 0,
                "correction_text_count": 0,
                "text_length_sum": 0.0,
                "source_text_length_sum": 0.0,
                "correction_text_length_sum": 0.0,
                "prototype_sums": {label: 0.0 for label in PROTOTYPE_TEXTS},
                "prototype_rows": 0,
                "topic_cluster_counts": {},
                "topic_distance_sum": 0.0,
                "embedding_mean_sum": 0.0,
                "embedding_std_sum": 0.0,
                "zero_shot_sums": {label: 0.0 for label in ZERO_SHOT_LABELS},
                "zero_shot_rows": 0,
            },
        )
        text_length = float(len(row.get("text") or ""))
        state["text_length_sum"] += text_length
        if row.get("post_type") == "source":
            state["source_text_count"] += 1
            state["source_text_length_sum"] += text_length
        else:
            state["correction_text_count"] += 1
            state["correction_text_length_sum"] += text_length

        feature_row = feature_lookup.get(str(row["post_id"]))
        if not feature_row:
            continue
        state["prototype_rows"] += 1
        state["embedding_mean_sum"] += float(feature_row.get("embedding_mean") or 0.0)
        state["embedding_std_sum"] += float(feature_row.get("embedding_std") or 0.0)
        state["topic_distance_sum"] += float(feature_row.get("topic_distance_to_center") or 0.0)
        cluster = str(feature_row.get("topic_cluster") or "0")
        state["topic_cluster_counts"][cluster] = state["topic_cluster_counts"].get(cluster, 0) + 1
        for label in PROTOTYPE_TEXTS:
            state["prototype_sums"][label] += float(feature_row.get(f"sim_{label}") or 0.0)
        if include_zero_shot:
            has_any_zs = False
            for label in ZERO_SHOT_LABELS:
                value = feature_row.get(f"zs_{label}")
                if value is not None and value == value:
                    state["zero_shot_sums"][label] += float(value)
                    has_any_zs = True
            if has_any_zs:
                state["zero_shot_rows"] += 1

    output_rows: list[dict[str, Any]] = []
    for thread_id, state in sorted(thread_state.items()):
        prototype_rows = max(int(state["prototype_rows"]), 1)
        zero_shot_rows = max(int(state["zero_shot_rows"]), 1)
        dominant_topic_cluster = "0"
        if state["topic_cluster_counts"]:
            dominant_topic_cluster = max(
                sorted(state["topic_cluster_counts"]),
                key=lambda cluster: state["topic_cluster_counts"][cluster],
            )
        dominant_proto = max(
            sorted(PROTOTYPE_TEXTS),
            key=lambda label: state["prototype_sums"][label],
        )
        row = {
            "dataset": dataset_name,
            "thread_id": thread_id,
            "avg_text_length": round(
                state["text_length_sum"] / max(state["source_text_count"] + state["correction_text_count"], 1),
                4,
            ),
            "avg_source_text_length": round(
                state["source_text_length_sum"] / max(state["source_text_count"], 1),
                4,
            ),
            "avg_correction_text_length": round(
                state["correction_text_length_sum"] / max(state["correction_text_count"], 1),
                4,
            ),
            "source_text_count": int(state["source_text_count"]),
            "correction_text_count": int(state["correction_text_count"]),
            "source_embedding_mean": round(state["embedding_mean_sum"] / prototype_rows, 6),
            "source_embedding_std": round(state["embedding_std_sum"] / prototype_rows, 6),
            "source_topic_cluster": dominant_topic_cluster,
            "source_topic_distance": round(state["topic_distance_sum"] / prototype_rows, 6),
            "prototype_dominant_strategy": dominant_proto,
            **{
                f"sim_{label}_mean": round(state["prototype_sums"][label] / prototype_rows, 6)
                for label in PROTOTYPE_TEXTS
            },
        }
        if include_zero_shot:
            row["early_correction_strategy"] = max(
                sorted(ZERO_SHOT_LABELS),
                key=lambda label: state["zero_shot_sums"][label],
            ) if state["zero_shot_rows"] else None
            for label in ZERO_SHOT_LABELS:
                row[f"zs_{label}_mean"] = round(state["zero_shot_sums"][label] / zero_shot_rows, 6)
        else:
            row["early_correction_strategy"] = dominant_proto
        output_rows.append(row)

    return pd.DataFrame(output_rows)


def _iter_canonical_rows(canonical_csv_path: Path) -> list[dict[str, Any]]:
    with canonical_csv_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_post_feature_row(
    row: dict[str, Any],
    hashed: str,
    config_id: str,
    extras: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "dataset": row.get("dataset"),
        "thread_id": row.get("thread_id"),
        "post_id": row.get("post_id"),
        "post_type": row.get("post_type"),
        "text_hash": hashed,
        "cache_key": f"{config_id}:{hashed}",
        "config_id": config_id,
    }
    base.update(extras)
    return base


def _project_cached_row(row: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
    projected = cached.copy()
    projected["dataset"] = row.get("dataset")
    projected["thread_id"] = row.get("thread_id")
    projected["post_id"] = row.get("post_id")
    projected["post_type"] = row.get("post_type")
    return projected


def _post_feature_columns() -> list[str]:
    return [
        "dataset",
        "thread_id",
        "post_id",
        "post_type",
        "text_hash",
        "cache_key",
        "config_id",
        "embedding_mean",
        "embedding_std",
        "topic_cluster",
        "topic_distance_to_center",
        *[f"sim_{label}" for label in PROTOTYPE_TEXTS],
        *[f"zs_{label}" for label in ZERO_SHOT_LABELS],
    ]


def _ensure_post_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in _post_feature_columns():
        if column not in frame.columns:
            frame[column] = np.nan
    ordered = [column for column in _post_feature_columns() if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    return frame[ordered + extras]


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_text_model_device() -> str:
    return "cuda" if _torch_cuda_available() else "cpu"


def _fit_topic_clusters(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    cluster_count = min(8, len(embeddings))
    if cluster_count <= 1:
        return np.zeros(len(embeddings), dtype=int), np.zeros(len(embeddings), dtype=float)
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
    labels = model.fit_predict(embeddings)
    centers = model.cluster_centers_[labels]
    distances = np.linalg.norm(embeddings - centers, axis=1)
    return labels.astype(int), distances.astype(float)


def _is_early_correction_candidate(row: dict[str, Any]) -> bool:
    hours = _float_or_none(row.get("timestamp_relative_hours"))
    if hours is None or hours > 6.0:
        return False
    if row.get("post_type") == "correction":
        return True
    if _as_bool(row.get("explicit_correction")):
        return True
    if _as_bool(row.get("is_correction")):
        return True
    if _as_bool(row.get("inferred_correction")):
        return True
    correction_type = str(row.get("correction_type") or "").strip().lower()
    return correction_type not in {"", "none", "nan"}


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float_or_none(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _ZeroShotPairDataset:
    def __init__(self, rows: list[tuple[dict[str, Any], str]], labels: tuple[str, ...]) -> None:
        self.rows = rows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, hashed = self.rows[index]
        return {
            "row": row,
            "text_hash": hashed,
            "premise": (row.get("text") or "")[:2048],
            "hypotheses": [_build_hypothesis(label) for label in self.labels],
        }


def _collate_zero_shot_batch(items: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    premises = []
    hypotheses = []
    batch_rows = []
    for item in items:
        premises.extend([item["premise"]] * len(item["hypotheses"]))
        hypotheses.extend(item["hypotheses"])
        batch_rows.append(
            {
                "dataset": item["row"].get("dataset"),
                "post_id": item["row"].get("post_id"),
                "text_hash": item["text_hash"],
            }
        )
    encoded = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    return {
        "rows": batch_rows,
        "encoded": encoded,
    }


def _build_hypothesis(label: str) -> str:
    label_to_text = {
        "denial": "This text is a denial-style correction.",
        "evidence_explanation": "This text provides evidence or factual explanation.",
        "action_guidance": "This text gives action guidance or advice.",
        "authority_statement": "This text is an authority or official statement.",
        "query": "This text mainly asks for verification or raises doubt.",
    }
    return label_to_text[label]


def _resolve_entailment_index(model) -> int:
    label2id = getattr(model.config, "label2id", {}) or {}
    for label, index in label2id.items():
        if label.lower().startswith("entail"):
            return int(index)
    return 2


def _resolve_contradiction_index(model) -> int | None:
    label2id = getattr(model.config, "label2id", {}) or {}
    for label, index in label2id.items():
        if label.lower().startswith("contrad"):
            return int(index)
    return 0 if getattr(model.config, "num_labels", 0) >= 3 else None
