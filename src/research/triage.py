from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io_utils import dump_json, ensure_dir


OBSERVATION_WINDOWS_HOURS = (1, 6, 24)
BUDGET_FRACTIONS = (0.05, 0.10, 0.20)
SEEDS = (11, 13, 17, 23, 29, 37, 42, 53, 71, 101)
PRIMARY_BUDGET = 0.10
COLD_START_QUANTILE = 0.50
HIGH_GROWTH_QUANTILE = 0.80
PRIMARY_VARIANT = "combined_catboost_ranker"
MAIN_METHOD_VARIANT = "combined_catboost_ranker"
SIGNIFICANCE_PRIMARY_VARIANTS = (
    "combined_xgb_ranker",
    "combined_catboost_ranker",
)
TEXT_MODEL_NAME = "xlm-roberta-base"
TEXT_MAX_CHARS = 6000
TEXT_MAX_LENGTH = 192
TEXT_BATCH_SIZE = 16
TEXT_EPOCHS = 1
TEXT_PAIRWISE_MAX_PAIRS = 2048
TRIAGE_SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRIAGE_SEMANTIC_BATCH_SIZE = 64
TRIAGE_SEMANTIC_MAX_CHARS = 6000
RANKNET_EPOCHS = 20
RANKNET_MAX_PAIRS = 4096
TREE_ESTIMATORS = 100
RANKER_ESTIMATORS = 80

MAIN_TABLE_TABULAR_MODEL_SPECS = (
    ("combined", "ridge"),
    ("combined", "rf"),
    ("response", "xgb"),
    ("semantic", "xgb"),
    ("cascade", "catboost"),
    ("combined", "xgb"),
    ("combined", "lgbm_lambdarank"),
    ("combined", "xgb_ranker"),
    ("combined", "catboost"),
)
MAIN_TABLE_TEXT_MODEL_KINDS = ("xlmr_pairwise",)
REPRESENTATIVE_VARIANTS = {
    "random_budget",
    "early_volume",
    "early_growth_rate",
    "recent_acceleration",
    "text_xlmr_pairwise_ranker",
    "combined_ridge_ranker",
    "combined_rf_ranker",
    "response_xgb_ranker",
    "semantic_xgb_ranker",
    "cascade_catboost_ranker",
    "combined_xgb_ranker",
    "combined_lgbm_lambdamart_ranker",
    "combined_xgb_ltr_ranker",
    "combined_catboost_ranker",
    "oracle_upper_bound",
}

# Disabled exploratory baselines from earlier runs. They are intentionally kept
# here as documentation but are not executed for the current main-table run.
LEGACY_MASS_RANKERS = (
    ("response", "response_mass_ranker"),
    ("semantic", "source_semantic_mass_ranker"),
    ("combined", "combined_mass_ranker"),
)
REGRESSION_MODEL_KINDS = ("ridge", "lasso", "elasticnet", "rf", "hgb", "xgb", "lgbm", "catboost")
REGRESSION_FEATURE_SETS = ("response", "semantic", "combined", "cascade")
LTR_MODEL_KINDS = ("lgbm_lambdarank", "xgb_ranker", "catboost_ranker", "ranknet")
LTR_FEATURE_SETS = ("response", "combined", "cascade")
TEXT_MODEL_KINDS = ("xlmr_regression", "xlmr_pairwise")
LATE_FUSION_VARIANT = "text_xlmr_response_cascade_fusion"
REGRESSION_BASELINE_SPECS = (
    *[(feature_set, model_kind) for feature_set in REGRESSION_FEATURE_SETS for model_kind in ("ridge", "lasso", "elasticnet", "rf", "hgb", "xgb")],
    ("combined", "lgbm"),
    ("cascade", "lgbm"),
    ("combined", "catboost"),
    ("cascade", "catboost"),
)
LTR_BASELINE_SPECS = (
    ("combined", "lgbm_lambdarank"),
    ("cascade", "lgbm_lambdarank"),
    ("combined", "xgb_ranker"),
    ("cascade", "xgb_ranker"),
    ("combined", "catboost_ranker"),
    ("cascade", "catboost_ranker"),
    ("combined", "ranknet"),
    ("cascade", "ranknet"),
)

RUMOR_LABELS = {"fake", "rumour", "rumor", "false", "misinformation"}
SOURCE_SEMANTIC_COLUMNS = (
    "embedding_mean",
    "embedding_std",
    "topic_cluster",
    "topic_distance_to_center",
    "sim_denial",
    "sim_evidence_explanation",
    "sim_action_guidance",
    "sim_authority_statement",
    "sim_emotion_reassurance",
)
SEMANTIC_CATEGORICAL_COLUMNS = (
    "source_sem_topic_cluster",
    "thread_text_source_topic_cluster",
    "thread_text_prototype_dominant_strategy",
    "thread_text_early_correction_strategy",
)


def run_triage_analysis(
    features_csv_path: Path,
    dataset_name: str,
    triage_dir: Path,
    *,
    text_model_dir: Path | None = None,
    canonical_csv_path: Path | None = None,
    model_root: Path | None = None,
    observation_windows: tuple[int, ...] = OBSERVATION_WINDOWS_HOURS,
    budget_fractions: tuple[float, ...] = BUDGET_FRACTIONS,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Evaluate annotation-free early intervention prioritization.

    The target is observed future cascade growth after an observation window.
    A good triage score should place a small intervention budget on threads
    that account for a large fraction of future spread.
    """
    ensure_dir(triage_dir)
    df = pd.read_csv(features_csv_path)
    if text_model_dir is not None:
        df = _merge_source_semantic_features(df, text_model_dir)
    if canonical_csv_path is not None and canonical_csv_path.exists():
        df = _merge_window_text_features(df, canonical_csv_path, observation_windows)
        df = _merge_window_semantic_features(
            df,
            observation_windows,
            cache_csv_path=triage_dir / "triage_semantic_features.csv",
        )

    rows: list[dict[str, Any]] = []
    resolved_model_root = ensure_dir(model_root or (triage_dir / "models"))
    for cohort_name, cohort_df in _iter_cohorts(df):
        if len(cohort_df) < 20:
            continue
        for window in observation_windows:
            if f"window_{window}h_reactions" not in cohort_df.columns:
                continue
            rows.extend(
                _run_window_triage(
                    cohort_df,
                    dataset_name=dataset_name,
                    cohort_name=cohort_name,
                    window=window,
                    budget_fractions=budget_fractions,
                    seeds=seeds,
                    model_root=resolved_model_root / cohort_name / f"{window}h",
                )
            )

    _assert_only_representative_variants(rows)
    results_csv = triage_dir / "triage_results.csv"
    _write_frame(results_csv, pd.DataFrame(rows))
    summary = _build_summary(dataset_name, rows)
    summary["triage_results_csv"] = str(results_csv)
    summary_path = triage_dir / "triage_summary.json"
    dump_json(summary_path, summary)
    return summary


def _run_window_triage(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    cohort_name: str,
    window: int,
    budget_fractions: tuple[float, ...],
    seeds: tuple[int, ...],
    model_root: Path,
) -> list[dict[str, Any]]:
    working = df.copy().reset_index(drop=True)
    observed = _numeric(working, f"window_{window}h_reactions", 0.0)
    total = _numeric(working, "reaction_total", 0.0)
    working["future_growth_target"] = (total - observed).clip(lower=0.0)
    if float(working["future_growth_target"].sum()) <= 0:
        return []

    high_growth = _high_growth_labels(working["future_growth_target"])
    if high_growth.nunique() < 2:
        return []
    working["high_future_growth"] = high_growth
    working["cold_start_thread"] = _cold_start_labels(observed)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_idx, test_idx = _split_indices(working, high_growth, seed)
        train_df = working.loc[train_idx].reset_index(drop=True)
        test_df = working.loc[test_idx].reset_index(drop=True)

        score_payloads = _baseline_scores(test_df, window, seed)
        score_payloads.extend(
            _model_scores(
                train_df,
                test_df,
                window,
                budget_fractions,
                seed,
                model_root,
                include_neural_text=True,
            )
        )
        score_payloads.append(
            {
                "variant": "oracle_upper_bound",
                "feature_set": "oracle",
                "model_family": "oracle",
                "model_name": "oracle_upper_bound",
                "training_objective": "observed_future_growth",
                "scores": test_df["future_growth_target"].astype(float).to_numpy(),
            }
        )

        y_future = test_df["future_growth_target"].astype(float).to_numpy()
        y_high = test_df["high_future_growth"].astype(int).to_numpy()
        for payload in score_payloads:
            scores = _safe_scores(payload["scores"], len(test_df))
            ranking_metrics = _ranking_metrics(y_future, y_high, scores)
            for budget in budget_fractions:
                budget_metrics = _budget_metrics(y_future, y_high, scores, budget)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "cohort": cohort_name,
                        "observation_window_h": window,
                        "seed": seed,
                        "variant": payload["variant"],
                        "feature_set": payload["feature_set"],
                        "model_family": payload.get("model_family"),
                        "model_name": payload.get("model_name"),
                        "training_objective": payload.get("training_objective"),
                        "test_threads": int(len(test_df)),
                        "future_growth_total": round(float(y_future.sum()), 4),
                        "high_growth_rate": round(float(y_high.mean()), 4),
                        "budget_fraction": budget,
                        "selected_base_variant": payload.get("selected_base_variant"),
                        "fusion_validation_capture": payload.get("validation_capture"),
                        "evaluation_slice": "all",
                        **ranking_metrics,
                        **budget_metrics,
                    }
                )
                cold_mask = test_df["cold_start_thread"].astype(bool).to_numpy()
                if cold_mask.any() and float(y_future[cold_mask].sum()) > 0:
                    cold_ranking_metrics = _ranking_metrics(
                        y_future[cold_mask],
                        y_high[cold_mask],
                        scores[cold_mask],
                    )
                    cold_budget_metrics = _budget_metrics(
                        y_future[cold_mask],
                        y_high[cold_mask],
                        scores[cold_mask],
                        budget,
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "cohort": cohort_name,
                            "observation_window_h": window,
                            "seed": seed,
                            "variant": payload["variant"],
                            "feature_set": payload["feature_set"],
                            "model_family": payload.get("model_family"),
                            "model_name": payload.get("model_name"),
                            "training_objective": payload.get("training_objective"),
                            "test_threads": int(cold_mask.sum()),
                            "future_growth_total": round(float(y_future[cold_mask].sum()), 4),
                            "high_growth_rate": round(float(y_high[cold_mask].mean()), 4),
                            "budget_fraction": budget,
                            "selected_base_variant": payload.get("selected_base_variant"),
                            "fusion_validation_capture": payload.get("validation_capture"),
                            "evaluation_slice": "cold_start",
                            **cold_ranking_metrics,
                            **cold_budget_metrics,
                        }
                    )
    return rows


def _iter_cohorts(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    cohorts = [("all_threads", df.copy())]
    if "thread_label" in df.columns:
        label = df["thread_label"].astype(str).str.lower()
        rumor_df = df[label.isin(RUMOR_LABELS)].copy()
        if not rumor_df.empty and len(rumor_df) != len(df):
            cohorts.append(("rumor_only", rumor_df))
        elif not rumor_df.empty:
            cohorts.append(("rumor_only", rumor_df))
    return cohorts


def _assert_only_representative_variants(rows: list[dict[str, Any]]) -> None:
    variants = {str(row.get("variant")) for row in rows if row.get("variant") is not None}
    unexpected = sorted(variants - REPRESENTATIVE_VARIANTS)
    if unexpected:
        raise RuntimeError(
            "Unexpected legacy triage variants produced: "
            + ", ".join(unexpected)
            + ". Current experiments must only run representative EBRT rankers."
        )


def _merge_source_semantic_features(df: pd.DataFrame, text_model_dir: Path) -> pd.DataFrame:
    merged = df.copy()
    post_path = text_model_dir / "post_semantic_features.csv"
    if post_path.exists() and post_path.stat().st_size > 0:
        available = pd.read_csv(post_path, nrows=0).columns
        usecols = [
            column
            for column in ("dataset", "thread_id", "post_id", "post_type", *SOURCE_SEMANTIC_COLUMNS)
            if column in available
        ]
        if {"dataset", "thread_id", "post_type"}.issubset(set(usecols)):
            semantic = pd.read_csv(post_path, usecols=usecols)
            semantic = semantic[semantic["post_type"].astype(str) == "source"].copy()
            if not semantic.empty:
                numeric_cols = [
                    column
                    for column in SOURCE_SEMANTIC_COLUMNS
                    if column in semantic.columns and column != "topic_cluster"
                ]
                grouped = (
                    semantic.groupby(["dataset", "thread_id"], as_index=False)[numeric_cols]
                    .mean(numeric_only=True)
                    .rename(columns={column: f"source_sem_{column}" for column in numeric_cols})
                )
                if "topic_cluster" in semantic.columns:
                    cluster = (
                        semantic.groupby(["dataset", "thread_id"])["topic_cluster"]
                        .agg(_mode_value)
                        .reset_index(name="source_sem_topic_cluster")
                    )
                    grouped = grouped.merge(cluster, on=["dataset", "thread_id"], how="left")
                merged = _merge_thread_level_features(merged, grouped)

    thread_path = text_model_dir / "thread_text_features.csv"
    if thread_path.exists() and thread_path.stat().st_size > 0:
        thread_text = pd.read_csv(thread_path)
        if not thread_text.empty and {"dataset", "thread_id"}.issubset(thread_text.columns):
            keep_cols = [
                column
                for column in thread_text.columns
                if column in {"dataset", "thread_id"}
                or column.startswith("sim_")
                or column.startswith("zs_")
                or column
                in {
                    "avg_source_text_length",
                    "source_topic_cluster",
                    "source_topic_distance",
                    "prototype_dominant_strategy",
                    "early_correction_strategy",
                }
            ]
            thread_text = thread_text[keep_cols].copy()
            rename_map = {
                column: f"thread_text_{column}"
                for column in thread_text.columns
                if column not in {"dataset", "thread_id"}
            }
            thread_text = thread_text.rename(columns=rename_map)
            merged = _merge_thread_level_features(merged, thread_text)

    return merged


def _merge_thread_level_features(df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or "thread_id" not in feature_df.columns:
        return df
    thread_key = "_triage_thread_id_key"
    left = df.copy()
    right = feature_df.copy()
    left[thread_key] = left["thread_id"].astype(str)
    right[thread_key] = right["thread_id"].astype(str)
    merge_keys = [thread_key]
    if "dataset" in left.columns and "dataset" in right.columns:
        merge_keys.insert(0, "dataset")
    right = right.drop(columns=["thread_id"])
    merged = left.merge(right, on=merge_keys, how="left")
    return merged.drop(columns=[thread_key])


def _merge_window_text_features(
    df: pd.DataFrame,
    canonical_csv_path: Path,
    observation_windows: tuple[int, ...],
) -> pd.DataFrame:
    thread_key = "_triage_thread_id_key"
    text_by_thread: dict[str, dict[int, list[str]]] = {}
    source_by_thread: dict[str, list[str]] = {}
    with canonical_csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            thread_id = str(row.get("thread_id") or "")
            if not thread_id:
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            post_type = str(row.get("post_type") or "")
            if post_type == "source":
                source_by_thread.setdefault(thread_id, []).append(text)
                continue
            hours = _float_or_none(row.get("timestamp_relative_hours"))
            if hours is None or hours < 0:
                continue
            bucket = text_by_thread.setdefault(thread_id, {window: [] for window in observation_windows})
            for window in observation_windows:
                if hours <= window:
                    bucket[window].append(text)

    rows: list[dict[str, Any]] = []
    for thread_id in df["thread_id"].astype(str):
        row: dict[str, Any] = {thread_key: thread_id}
        source_text = source_by_thread.get(thread_id, [])
        by_window = text_by_thread.get(thread_id, {})
        for window in observation_windows:
            parts = [*source_text, *by_window.get(window, [])]
            row[f"window_{window}h_text"] = _truncate_text(" ".join(parts), TEXT_MAX_CHARS)
        rows.append(row)
    text_df = pd.DataFrame(rows)
    merged = df.copy()
    merged[thread_key] = merged["thread_id"].astype(str)
    merged = merged.merge(text_df, on=thread_key, how="left")
    return merged.drop(columns=[thread_key])


def _merge_window_semantic_features(
    df: pd.DataFrame,
    observation_windows: tuple[int, ...],
    *,
    cache_csv_path: Path,
) -> pd.DataFrame:
    if not _triage_semantic_features_enabled():
        return df
    required_text_columns = [f"window_{window}h_text" for window in observation_windows]
    if not any(column in df.columns for column in required_text_columns):
        return df

    cached = _read_cached_window_semantics(cache_csv_path, df, observation_windows)
    if cached is None:
        cached = _build_window_semantic_features(df, observation_windows)
        if cached.empty:
            return df
        ensure_dir(cache_csv_path.parent)
        cached.to_csv(cache_csv_path, index=False)
    return _merge_thread_level_features(df, cached)


def _read_cached_window_semantics(
    cache_csv_path: Path,
    df: pd.DataFrame,
    observation_windows: tuple[int, ...],
) -> pd.DataFrame | None:
    if not cache_csv_path.exists() or cache_csv_path.stat().st_size <= 0:
        return None
    try:
        cached = pd.read_csv(cache_csv_path)
    except Exception:
        return None
    if cached.empty or "thread_id" not in cached.columns:
        return None
    required_columns = {
        "thread_id",
        *[
            f"window_{window}h_sem_sim_denial"
            for window in observation_windows
        ],
    }
    if not required_columns.issubset(cached.columns):
        return None
    current_ids = set(df["thread_id"].astype(str))
    cached_ids = set(cached["thread_id"].astype(str))
    if not current_ids.issubset(cached_ids):
        return None
    return cached


def _build_window_semantic_features(
    df: pd.DataFrame,
    observation_windows: tuple[int, ...],
) -> pd.DataFrame:
    try:
        from sentence_transformers import SentenceTransformer

        from .text_models import PROTOTYPE_TEXTS
    except Exception:
        return pd.DataFrame()

    text_columns = [f"window_{window}h_text" for window in observation_windows if f"window_{window}h_text" in df.columns]
    if not text_columns:
        return pd.DataFrame()

    model_name = _triage_semantic_model_name()
    device = _resolve_torch_device()
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        model = SentenceTransformer(model_name, device="cpu")

    prototype_labels = list(PROTOTYPE_TEXTS)
    prototype_embeddings = model.encode(
        [PROTOTYPE_TEXTS[label] for label in prototype_labels],
        batch_size=min(_triage_semantic_batch_size(), len(prototype_labels)),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    base_columns: dict[str, Any] = {"thread_id": df["thread_id"].astype(str).to_numpy()}
    if "dataset" in df.columns:
        base_columns["dataset"] = df["dataset"].astype(str).to_numpy()
    feature_blocks = [pd.DataFrame(base_columns, index=df.index)]

    for window in observation_windows:
        text_column = f"window_{window}h_text"
        if text_column not in df.columns:
            continue
        texts = (
            df[text_column]
            .fillna("")
            .astype(str)
            .map(lambda value: _truncate_text(value, _triage_semantic_max_chars()))
            .tolist()
        )
        non_empty = np.asarray([bool(text.strip()) for text in texts], dtype=bool)
        embeddings = np.zeros((len(texts), prototype_embeddings.shape[1]), dtype=np.float32)
        if non_empty.any():
            encoded = model.encode(
                [text for text, keep in zip(texts, non_empty) if keep],
                batch_size=_triage_semantic_batch_size(),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            embeddings[non_empty] = encoded

        similarities = embeddings @ prototype_embeddings.T
        feature_columns: dict[str, Any] = {
            f"window_{window}h_sem_text_length": [len(text) for text in texts],
            f"window_{window}h_sem_embedding_mean": embeddings.mean(axis=1),
            f"window_{window}h_sem_embedding_std": embeddings.std(axis=1),
        }
        for label_index, label in enumerate(prototype_labels):
            feature_columns[f"window_{window}h_sem_sim_{label}"] = similarities[:, label_index]
        for dim_index in range(embeddings.shape[1]):
            feature_columns[f"window_{window}h_sem_emb_{dim_index:03d}"] = embeddings[:, dim_index]
        feature_blocks.append(pd.DataFrame(feature_columns, index=df.index))

    return pd.concat(feature_blocks, axis=1).copy()


def _resolve_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _mode_value(values: pd.Series) -> str | None:
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return None
    return str(cleaned.value_counts().sort_index().idxmax())


def _baseline_scores(test_df: pd.DataFrame, window: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    early_growth_rate = _numeric(test_df, f"window_{window}h_growth_rate", 0.0).to_numpy()
    recent_acceleration = _recent_acceleration_scores(test_df, window)
    return [
        {
            "variant": "random_budget",
            "feature_set": "random",
            "model_family": "heuristic",
            "model_name": "random",
            "training_objective": "none",
            "scores": rng.random(len(test_df)),
        },
        {
            "variant": "early_volume",
            "feature_set": "volume",
            "model_family": "heuristic",
            "model_name": "early_volume",
            "training_objective": "none",
            "scores": _numeric(test_df, f"window_{window}h_reactions", 0.0).to_numpy(),
        },
        {
            "variant": "early_growth_rate",
            "feature_set": "volume",
            "model_family": "heuristic",
            "model_name": "early_growth_rate",
            "training_objective": "none",
            "scores": early_growth_rate,
        },
        {
            "variant": "recent_acceleration",
            "feature_set": "volume",
            "model_family": "heuristic",
            "model_name": "recent_acceleration",
            "training_objective": "none",
            "scores": recent_acceleration,
        },
    ]


def _model_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int,
    budget_fractions: tuple[float, ...],
    seed: int,
    model_root: Path,
    include_neural_text: bool,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str], np.ndarray] = {}
    ensure_dir(model_root)
    # Disabled for the current main-table run:
    # - budgeted model selector / late fusion
    # - legacy mass-linear diagnostic rankers
    # - Ridge/Lasso/ElasticNet/RF/HGB/LGBM broad sweeps
    # - broad LTR/RankNet exploratory sweeps beyond the direct combined XGBRanker baseline
    # Only the representative main-table rows below are executed.
    for feature_set, model_kind in MAIN_TABLE_TABULAR_MODEL_SPECS:
        scores = _cached_ranker_scores(score_cache, train_df, test_df, window, feature_set, model_kind, seed, model_root)
        if scores is None:
            continue
        training_objective = (
            _ltr_objective(model_kind)
            if model_kind in LTR_MODEL_KINDS
            else "log_future_growth_regression"
        )
        payloads.append(
            {
                "variant": _variant_name(feature_set, model_kind),
                "feature_set": feature_set,
                "model_family": _model_family(model_kind),
                "model_name": model_kind,
                "training_objective": training_objective,
                "scores": scores,
            }
        )
    if include_neural_text and _text_baselines_enabled():
        for model_kind in MAIN_TABLE_TEXT_MODEL_KINDS:
            scores = _cached_ranker_scores(score_cache, train_df, test_df, window, "text", model_kind, seed, model_root)
            if scores is None:
                continue
            payloads.append(
                {
                    "variant": _variant_name("text", model_kind),
                    "feature_set": "text",
                    "model_family": "neural_text",
                    "model_name": model_kind,
                    "training_objective": _text_objective(model_kind),
                    "scores": scores,
                }
            )
    return payloads


def _cached_ranker_scores(
    score_cache: dict[tuple[str, str], np.ndarray],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int,
    feature_set: str,
    model_kind: str,
    seed: int,
    model_root: Path,
) -> np.ndarray | None:
    key = (feature_set, model_kind)
    if key not in score_cache:
        scores = _fit_ranker_scores(
            train_df,
            test_df,
            window,
            feature_set,
            model_kind,
            seed=seed,
            model_root=model_root / feature_set / model_kind,
        )
        if scores is None:
            return None
        score_cache[key] = scores
    return score_cache[key]


def _diagnostic_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int,
    seed: int,
    score_cache: dict[tuple[str, str], np.ndarray],
    model_root: Path,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    response_scores = _cached_ranker_scores(score_cache, train_df, test_df, window, "response", "mass_linear", seed, model_root)
    semantic_scores = _cached_ranker_scores(score_cache, train_df, test_df, window, "semantic", "mass_linear", seed, model_root)
    combined_scores = _cached_ranker_scores(score_cache, train_df, test_df, window, "combined", "mass_linear", seed, model_root)
    residual_scores = _semantic_residual_scores(
        train_df,
        test_df,
        window,
        response_scores=response_scores,
        semantic_scores=semantic_scores,
    )
    if residual_scores is not None:
        payloads.append(
            {
                "variant": "semantic_residual_ranker",
                "feature_set": "semantic_residual",
                "model_family": "diagnostic",
                "model_name": "response_plus_semantic_residual",
                "training_objective": "posthoc_score_fusion",
                "scores": residual_scores,
            }
    )
    for feature_group, variant in (
        ("semantic", "combined_permute_semantic"),
        ("response", "combined_permute_response"),
    ):
        scores = _permuted_proxy_scores(
            combined_scores,
            test_df,
            window,
            feature_group=feature_group,
            seed=seed,
        )
        if scores is not None:
            payloads.append(
                {
                    "variant": variant,
                    "feature_set": f"combined_permuted_{feature_group}",
                    "model_family": "diagnostic",
                    "model_name": "group_permutation_proxy",
                    "training_objective": "diagnostic_permutation",
                    "scores": scores,
                }
            )
    return payloads


def _budgeted_model_selector_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int,
    budget_fractions: tuple[float, ...],
    model_root: Path,
) -> dict[str, Any] | None:
    fit_df, valid_df = _validation_split(train_df)
    candidates: list[dict[str, Any]] = [
        {
            "name": "early_growth_rate",
            "valid_scores": _numeric(valid_df, f"window_{window}h_growth_rate", 0.0).to_numpy(),
            "test_scores": _numeric(test_df, f"window_{window}h_growth_rate", 0.0).to_numpy(),
        }
    ]
    candidates.append(
        {
            "name": "early_volume",
            "valid_scores": _numeric(valid_df, f"window_{window}h_reactions", 0.0).to_numpy(),
            "test_scores": _numeric(test_df, f"window_{window}h_reactions", 0.0).to_numpy(),
        }
    )
    selector_candidates = [
        *[(variant, feature_set, "mass_linear") for feature_set, variant in LEGACY_MASS_RANKERS],
        ("combined_ridge_ranker", "combined", "ridge"),
        ("combined_xgb_ranker", "combined", "xgb"),
        ("combined_lgbm_ranker", "combined", "lgbm"),
        ("combined_catboost_ranker", "combined", "catboost"),
        ("response_xgb_ranker", "response", "xgb"),
        ("cascade_xgb_ranker", "cascade", "xgb"),
        ("cascade_lgbm_lambdarank_ranker", "cascade", "lgbm_lambdarank"),
        ("combined_lgbm_lambdarank_ranker", "combined", "lgbm_lambdarank"),
    ]
    for name, feature_set, model_kind in selector_candidates:
        valid_scores = _fit_ranker_scores(
            fit_df,
            valid_df,
            window,
            feature_set,
            model_kind,
            seed=7,
            model_root=model_root / "selector" / feature_set / model_kind / "valid",
        )
        test_scores = _fit_ranker_scores(
            fit_df,
            test_df,
            window,
            feature_set,
            model_kind,
            seed=7,
            model_root=model_root / "selector" / feature_set / model_kind / "test",
        )
        if valid_scores is None or test_scores is None:
            continue
        candidates.append(
            {
                "name": name,
                "valid_scores": valid_scores,
                "test_scores": test_scores,
            }
        )
    if len(candidates) < 2:
        return None

    valid_future = valid_df["future_growth_target"].astype(float).to_numpy()
    best_candidate = max(
        candidates,
        key=lambda candidate: np.mean(
            [
                _capture_rate(valid_future, candidate["valid_scores"], budget_fraction)
                for budget_fraction in budget_fractions
            ]
        ),
    )
    best_score = float(
        np.mean(
            [
                _capture_rate(valid_future, best_candidate["valid_scores"], budget_fraction)
                for budget_fraction in budget_fractions
            ]
        )
    )
    return {
        "variant": PRIMARY_VARIANT,
        "feature_set": "validated_model_selection",
        "model_family": "model_selection",
        "model_name": best_candidate["name"],
        "training_objective": "validation_capture_selection",
        "scores": best_candidate["test_scores"],
        "selected_base_variant": best_candidate["name"],
        "validation_capture": round(best_score, 4),
    }


def _fit_ranker_scores(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    window: int,
    feature_set: str,
    model_kind: str,
    *,
    permute_feature_group: str | None = None,
    seed: int = 42,
    model_root: Path | None = None,
) -> np.ndarray | None:
    if feature_set == "text":
        return _fit_text_ranker_scores(train_df, score_df, window, model_kind, seed, model_root)
    X_train = _feature_frame(train_df, window, feature_set)
    X_score = _feature_frame(score_df, window, feature_set)
    if X_train.empty or X_score.empty:
        return None
    X_score = X_score.reindex(columns=X_train.columns, fill_value=0.0)
    if permute_feature_group is not None:
        X_score = _permute_feature_group(
            X_score,
            feature_group=permute_feature_group,
            seed=seed,
        )
    y_train = np.log1p(train_df["future_growth_target"].astype(float).to_numpy())
    sample_weight = 1.0 + y_train
    if model_kind == "mass_linear":
        return _linear_ranker_scores(X_train, X_score, y_train, sample_weight)
    if model_kind in {"ridge", "lasso", "elasticnet"}:
        return _sklearn_linear_scores(X_train, X_score, y_train, sample_weight, model_kind, seed)
    if model_kind == "rf":
        model = RandomForestRegressor(
            n_estimators=TREE_ESTIMATORS,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model.predict(X_score)
    if model_kind == "hgb":
        model = HistGradientBoostingRegressor(
            random_state=seed,
            l2_regularization=0.01,
            max_iter=TREE_ESTIMATORS,
            learning_rate=0.05,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model.predict(X_score)
    if model_kind == "xgb":
        return _xgb_regressor_scores(X_train, X_score, y_train, sample_weight, seed)
    if model_kind == "lgbm":
        return _lgbm_regressor_scores(X_train, X_score, y_train, sample_weight, seed)
    if model_kind == "catboost":
        return _catboost_regressor_scores(X_train, X_score, y_train, sample_weight, seed)
    if model_kind == "lgbm_lambdarank":
        return _lgbm_ranker_scores(X_train, X_score, y_train, seed)
    if model_kind == "xgb_ranker":
        return _xgb_ranker_scores(X_train, X_score, y_train, seed)
    if model_kind == "catboost_ranker":
        return _catboost_ranker_scores(X_train, X_score, y_train, seed)
    if model_kind == "ranknet":
        return _ranknet_scores(X_train, X_score, y_train, seed)
    return None


def _sklearn_linear_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    model_kind: str,
    seed: int,
) -> np.ndarray:
    if model_kind == "ridge":
        estimator = Ridge(alpha=1.0, random_state=seed)
    elif model_kind == "lasso":
        estimator = Lasso(alpha=0.001, max_iter=10000, random_state=seed)
    else:
        estimator = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=seed)
    model = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    model.fit(X_train, y_train, model__sample_weight=sample_weight)
    return model.predict(X_score)


def _xgb_regressor_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None
    params = dict(
        n_estimators=TREE_ESTIMATORS,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
    )
    try:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    except Exception:
        params["device"] = "cpu"
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    return model.predict(X_score)


def _lgbm_regressor_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None
    params = dict(
        n_estimators=TREE_ESTIMATORS,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=4,
        force_col_wise=True,
        verbose=-1,
    )
    device_type = os.getenv("RUMOR_SIM_LGBM_DEVICE_TYPE")
    if device_type:
        params["device_type"] = device_type
    try:
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except Exception:
        params.pop("device_type", None)
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_score)


def _catboost_regressor_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    try:
        from catboost import CatBoostRegressor
    except Exception:
        return None
    params = dict(
        iterations=TREE_ESTIMATORS,
        depth=6,
        learning_rate=0.035,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",
    )
    try:
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except Exception:
        params.pop("task_type", None)
        params.pop("devices", None)
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_score)


def _lgbm_ranker_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    try:
        from lightgbm import LGBMRanker
    except Exception:
        return None
    relevance = _relevance_labels(y_train)
    if len(np.unique(relevance)) < 2:
        return None
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=RANKER_ESTIMATORS,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=8,
        random_state=seed,
        n_jobs=4,
        force_col_wise=True,
        verbose=-1,
    )
    device_type = os.getenv("RUMOR_SIM_LGBM_DEVICE_TYPE")
    if device_type:
        params["device_type"] = device_type
    try:
        model = LGBMRanker(**params)
        model.fit(X_train, relevance, group=[len(X_train)])
    except Exception:
        params.pop("device_type", None)
        model = LGBMRanker(**params)
        model.fit(X_train, relevance, group=[len(X_train)])
    return model.predict(X_score)


def _xgb_ranker_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    try:
        from xgboost import XGBRanker
    except Exception:
        return None
    relevance = _relevance_labels(y_train)
    if len(np.unique(relevance)) < 2:
        return None
    qid = np.zeros(len(X_train), dtype=int)
    params = dict(
        objective="rank:ndcg",
        n_estimators=RANKER_ESTIMATORS,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
    )
    try:
        model = XGBRanker(**params)
        model.fit(X_train, relevance, qid=qid, verbose=False)
    except Exception:
        params["device"] = "cpu"
        model = XGBRanker(**params)
        model.fit(X_train, relevance, qid=qid, verbose=False)
    return model.predict(X_score)


def _catboost_ranker_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    try:
        from catboost import CatBoostRanker, Pool
    except Exception:
        return None
    relevance = _relevance_labels(y_train)
    if len(np.unique(relevance)) < 2:
        return None
    sampled_pairs = _sample_pairs(y_train, min(4096, max(256, len(y_train) * 16)), np.random.default_rng(seed))
    if sampled_pairs is None:
        return None
    left, right, labels = sampled_pairs
    winners = np.where(labels > 0.5, left, right)
    losers = np.where(labels > 0.5, right, left)
    pairs = np.column_stack([winners, losers]).astype(int)
    group_id = np.zeros(len(X_train), dtype=int)
    score_group_id = np.zeros(len(X_score), dtype=int)
    params = dict(
        iterations=RANKER_ESTIMATORS,
        depth=6,
        learning_rate=0.04,
        loss_function="PairLogit",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",
    )
    train_pool = Pool(X_train, label=relevance, group_id=group_id, pairs=pairs)
    score_pool = Pool(X_score, group_id=score_group_id)
    try:
        model = CatBoostRanker(**params)
        model.fit(train_pool)
    except Exception:
        params.pop("task_type", None)
        params.pop("devices", None)
        model = CatBoostRanker(**params)
        model.fit(train_pool)
    return model.predict(score_pool)


def _ranknet_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    if len(np.unique(y_train)) < 2:
        return None
    try:
        import torch
        from torch import nn
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    train = X_train.to_numpy(dtype=np.float32)
    score = X_score.to_numpy(dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std <= 1e-6] = 1.0
    train = (train - mean) / std
    score = (score - mean) / std
    pairs = _sample_pairs(y_train, min(RANKNET_MAX_PAIRS, max(256, len(train) * 32)), rng)
    if pairs is None:
        return None
    left, right, labels = pairs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(train.shape[1], 96),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(96, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_tensor = torch.as_tensor(train, device=device)
    left_tensor = torch.as_tensor(left, dtype=torch.long, device=device)
    right_tensor = torch.as_tensor(right, dtype=torch.long, device=device)
    label_tensor = torch.as_tensor(labels.astype(np.float32), device=device)
    for _ in range(RANKNET_EPOCHS):
        order = torch.randperm(len(left_tensor), device=device)
        for start in range(0, len(order), 512):
            idx = order[start : start + 512]
            diff = model(train_tensor[left_tensor[idx]]).squeeze(-1) - model(train_tensor[right_tensor[idx]]).squeeze(-1)
            loss = loss_fn(diff, label_tensor[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        scores = model(torch.as_tensor(score, device=device)).squeeze(-1).detach().cpu().numpy()
    _clear_torch_cache()
    return scores


def _fit_text_ranker_scores(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    window: int,
    model_kind: str,
    seed: int,
    model_root: Path | None,
) -> np.ndarray | None:
    text_column = f"window_{window}h_text"
    if text_column not in train_df.columns or text_column not in score_df.columns:
        return None
    train_texts = train_df[text_column].fillna("").astype(str).tolist()
    score_texts = score_df[text_column].fillna("").astype(str).tolist()
    if sum(bool(text.strip()) for text in train_texts) < 4 or not any(text.strip() for text in score_texts):
        return None
    if model_kind == "xlmr_regression":
        return _fit_transformer_regression_scores(train_texts, score_texts, train_df, seed, model_root)
    if model_kind == "xlmr_pairwise":
        return _fit_transformer_pairwise_scores(train_texts, score_texts, train_df, seed, model_root)
    return None


def _fit_transformer_regression_scores(
    train_texts: list[str],
    score_texts: list[str],
    train_df: pd.DataFrame,
    seed: int,
    model_root: Path | None,
) -> np.ndarray | None:
    try:
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging
    except Exception:
        return None
    hf_logging.set_verbosity_error()
    y = np.log1p(train_df["future_growth_target"].astype(float).to_numpy())
    if len(np.unique(y)) < 2:
        return None
    model_name = _text_model_name()
    max_length = _text_max_length()
    batch_size = _text_batch_size()
    epochs = _text_epochs()
    ensure_dir(model_root or Path("/tmp"))
    _ensure_hf_cache(model_root)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
    except Exception:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = _TransformerScorer(encoder).to(device)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0
    targets = torch.as_tensor(((y - y_mean) / y_std).astype(np.float32), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    order = np.arange(len(train_texts))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            encoded = _encode_text_batch(tokenizer, [train_texts[i] for i in idx], max_length, device)
            pred = model(**encoded)
            loss = loss_fn(pred, targets[torch.as_tensor(idx, dtype=torch.long, device=device)])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    scores = _predict_transformer_scores(model, tokenizer, score_texts, max_length, batch_size, device)
    _clear_torch_cache()
    return scores


def _fit_transformer_pairwise_scores(
    train_texts: list[str],
    score_texts: list[str],
    train_df: pd.DataFrame,
    seed: int,
    model_root: Path | None,
) -> np.ndarray | None:
    try:
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging
    except Exception:
        return None
    hf_logging.set_verbosity_error()
    y = np.log1p(train_df["future_growth_target"].astype(float).to_numpy())
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    pairs = _sample_pairs(y, min(TEXT_PAIRWISE_MAX_PAIRS, max(256, len(y) * 24)), rng)
    if pairs is None:
        return None
    left, right, labels = pairs
    model_name = _text_model_name()
    max_length = _text_max_length()
    batch_size = _text_batch_size()
    epochs = _text_epochs()
    ensure_dir(model_root or Path("/tmp"))
    _ensure_hf_cache(model_root)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
    except Exception:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = _TransformerScorer(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    pair_order = np.arange(len(left))
    model.train()
    for _ in range(epochs):
        rng.shuffle(pair_order)
        for start in range(0, len(pair_order), batch_size):
            idx = pair_order[start : start + batch_size]
            left_encoded = _encode_text_batch(tokenizer, [train_texts[i] for i in left[idx]], max_length, device)
            right_encoded = _encode_text_batch(tokenizer, [train_texts[i] for i in right[idx]], max_length, device)
            diff = model(**left_encoded) - model(**right_encoded)
            target = torch.as_tensor(labels[idx].astype(np.float32), device=device)
            loss = loss_fn(diff, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    scores = _predict_transformer_scores(model, tokenizer, score_texts, max_length, batch_size, device)
    _clear_torch_cache()
    return scores


class _TransformerScorer:
    def __new__(cls, encoder: Any) -> Any:
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for transformer scoring") from exc

        class Scorer(nn.Module):
            def __init__(self, base: Any) -> None:
                super().__init__()
                self.base = base
                hidden_size = int(getattr(base.config, "hidden_size", 768))
                self.dropout = nn.Dropout(0.10)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, **encoded: Any) -> Any:
                output = self.base(**encoded)
                pooled = output.last_hidden_state[:, 0, :]
                return self.head(self.dropout(pooled)).squeeze(-1)

        return Scorer(encoder)


def _encode_text_batch(tokenizer: Any, texts: list[str], max_length: int, device: Any) -> dict[str, Any]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _predict_transformer_scores(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import torch

    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = _encode_text_batch(tokenizer, texts[start : start + batch_size], max_length, device)
            scores.append(model(**encoded).detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=float)


def _relevance_labels(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train, dtype=float)
    if len(np.unique(y)) < 2:
        return np.zeros(len(y), dtype=int)
    ranks = pd.Series(y).rank(method="average", pct=True).to_numpy()
    return np.clip(np.floor(ranks * 5), 0, 4).astype(int)


def _sample_pairs(
    y_train: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    y = np.asarray(y_train, dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    left: list[int] = []
    right: list[int] = []
    labels: list[float] = []
    attempts = 0
    max_attempts = max_pairs * 12
    while len(left) < max_pairs and attempts < max_attempts:
        i, j = rng.integers(0, len(y), size=2)
        attempts += 1
        if i == j or abs(float(y[i] - y[j])) <= 1e-12:
            continue
        left.append(int(i))
        right.append(int(j))
        labels.append(1.0 if y[i] > y[j] else 0.0)
    if len(left) < 8:
        return None
    return np.asarray(left), np.asarray(right), np.asarray(labels, dtype=float)


def _clear_torch_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ensure_hf_cache(model_root: Path | None) -> None:
    if model_root is None:
        return
    cache_root = _output_root_for_cache(model_root) / "hf_cache"
    ensure_dir(cache_root)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))


def _output_root_for_cache(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if parent.name == "outputs":
            return parent
    return resolved


def _linear_ranker_scores(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    train = X_train.to_numpy(dtype=float)
    score = X_score.to_numpy(dtype=float)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std <= 1e-12] = 1.0
    train = (train - mean) / std
    score = (score - mean) / std
    weights = np.asarray(sample_weight, dtype=float)
    y_mean = float(np.average(y_train, weights=weights))
    centered_y = y_train - y_mean
    numerator = (train * (centered_y * weights)[:, None]).sum(axis=0)
    denominator = ((train**2) * weights[:, None]).sum(axis=0) + 1e-6
    coef = numerator / denominator
    coef_norm = float(np.linalg.norm(coef))
    if coef_norm > 0:
        coef = coef / coef_norm
    return score @ coef


def _semantic_residual_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int,
    *,
    response_scores: np.ndarray | None,
    semantic_scores: np.ndarray | None,
) -> np.ndarray | None:
    if response_scores is None or semantic_scores is None:
        return None
    response_z = _zscore(response_scores)
    semantic_z = _zscore(semantic_scores)
    return response_z + 0.35 * semantic_z


def _permuted_proxy_scores(
    combined_scores: np.ndarray | None,
    test_df: pd.DataFrame,
    window: int,
    *,
    feature_group: str,
    seed: int,
) -> np.ndarray | None:
    if combined_scores is None:
        return None
    fallback_feature_set = "response" if feature_group == "semantic" else "semantic"
    fallback = _cheap_proxy_scores(test_df, window, fallback_feature_set)
    if fallback is None:
        return None
    rng = np.random.default_rng(seed)
    noise = rng.permutation(_zscore(combined_scores))
    return _zscore(fallback) + 0.05 * noise


def _cheap_proxy_scores(df: pd.DataFrame, window: int, feature_set: str) -> np.ndarray | None:
    if feature_set == "response":
        candidates = [
            f"window_{window}h_reactions",
            f"window_{window}h_growth_rate",
            f"window_{window}h_query_count",
            f"window_{window}h_evidence_count",
            f"window_{window}h_deny_count",
        ]
        available = [column for column in candidates if column in df.columns]
        if not available:
            return None
        score = np.zeros(len(df), dtype=float)
        for column in available:
            score += _zscore(_numeric(df, column, 0.0).to_numpy())
        return score
    if feature_set == "semantic":
        available = [
            column
            for column in df.columns
            if (
                column.startswith("source_sem_sim_")
                or column.startswith("thread_text_sim_")
                or column.startswith("thread_text_zs_")
            )
        ]
        if not available:
            return None
        score = np.zeros(len(df), dtype=float)
        for column in available:
            score += _zscore(_numeric(df, column, 0.0).to_numpy())
        return score
    return None


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    std = float(np.std(array))
    if std <= 1e-12:
        return np.zeros_like(array, dtype=float)
    return (array - float(np.mean(array))) / std


def _feature_frame(df: pd.DataFrame, window: int, feature_set: str) -> pd.DataFrame:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    if feature_set in {"response", "combined", "cascade"}:
        numeric_cols.extend(
            [
                "source_length",
                "source_has_url",
                "source_punctuation_intensity",
                "source_is_correction",
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
        categorical_cols.extend(["source_stance_type", "source_publisher_type"])
    if feature_set == "cascade":
        observable_windows = [candidate for candidate in OBSERVATION_WINDOWS_HOURS if candidate <= window]
        for candidate in observable_windows:
            numeric_cols.extend(
                [
                    f"window_{candidate}h_reactions",
                    f"window_{candidate}h_growth_rate",
                    f"window_{candidate}h_corrections",
                    f"window_{candidate}h_official_corrections",
                    f"window_{candidate}h_deny_count",
                    f"window_{candidate}h_query_count",
                    f"window_{candidate}h_evidence_count",
                    f"window_{candidate}h_emotion_count",
                    f"window_{candidate}h_other_count",
                ]
            )
    if feature_set in {"semantic", "combined"}:
        semantic_allowed_prefixes = [
            "source_sem_",
            f"window_{window}h_sem_",
        ]
        if window >= 6:
            semantic_allowed_prefixes.extend(
                [
                    "thread_text_sim_",
                    "thread_text_zs_",
                    "thread_text_source_topic_",
                    "thread_text_avg_source_text_length",
                ]
            )
        numeric_cols.extend(
            [
                column
                for column in df.columns
                if column.startswith(tuple(semantic_allowed_prefixes))
                and column not in SEMANTIC_CATEGORICAL_COLUMNS
            ]
        )
        if feature_set == "semantic":
            semantic_categorical = ["source_sem_topic_cluster"]
            if window >= 6:
                semantic_categorical.extend(
                    [
                        "thread_text_source_topic_cluster",
                        "thread_text_prototype_dominant_strategy",
                        "thread_text_early_correction_strategy",
                    ]
                )
            categorical_cols.extend([column for column in semantic_categorical if column in df.columns])

    numeric_cols = [column for column in numeric_cols if column in df.columns]
    categorical_cols = [column for column in categorical_cols if column in df.columns]
    if not numeric_cols and not categorical_cols:
        return pd.DataFrame(index=df.index)

    frame = pd.DataFrame(
        {column: _numeric(df, column, 0.0).to_numpy() for column in numeric_cols},
        index=df.index,
    )
    if feature_set == "cascade":
        frame["first_reaction_observed_hours"] = _observed_time(df, "first_reaction_hours", window)
        frame["first_correction_observed_hours"] = _observed_time(df, "first_correction_hours", window)
        frame["first_official_correction_observed_hours"] = _observed_time(df, "first_official_correction_hours", window)
        if window >= 6 and "window_1h_growth_rate" in frame.columns and "window_6h_growth_rate" in frame.columns:
            frame["cascade_growth_delta_1h_6h"] = frame["window_6h_growth_rate"] - frame["window_1h_growth_rate"]
            frame["cascade_reaction_ratio_1h_6h"] = _safe_ratio(frame["window_6h_reactions"], frame["window_1h_reactions"])
        if window >= 24 and "window_6h_growth_rate" in frame.columns and "window_24h_growth_rate" in frame.columns:
            frame["cascade_growth_delta_6h_24h"] = frame["window_24h_growth_rate"] - frame["window_6h_growth_rate"]
            frame["cascade_reaction_ratio_6h_24h"] = _safe_ratio(frame["window_24h_reactions"], frame["window_6h_reactions"])
    if categorical_cols:
        categorical = df[categorical_cols].fillna("missing").astype(str)
        frame = pd.concat([frame, pd.get_dummies(categorical, dummy_na=True)], axis=1)
    return frame.fillna(0.0)


def _high_growth_labels(target: pd.Series) -> pd.Series:
    positive = target[target > 0]
    if positive.empty:
        return pd.Series([0] * len(target), index=target.index)
    threshold = float(positive.quantile(HIGH_GROWTH_QUANTILE))
    return ((target >= threshold) & (target > 0)).astype(int)


def _cold_start_labels(observed: pd.Series) -> pd.Series:
    observed_numeric = pd.to_numeric(observed, errors="coerce").fillna(0.0)
    positive = observed_numeric[observed_numeric > 0]
    if positive.empty:
        return pd.Series([True] * len(observed_numeric), index=observed_numeric.index)
    threshold = float(positive.quantile(COLD_START_QUANTILE))
    return (observed_numeric <= threshold).astype(bool)


def _recent_acceleration_scores(df: pd.DataFrame, window: int) -> np.ndarray:
    current = _numeric(df, f"window_{window}h_growth_rate", 0.0)
    previous_window = max(
        (candidate for candidate in OBSERVATION_WINDOWS_HOURS if candidate < window),
        default=None,
    )
    if previous_window is None:
        return current.to_numpy()
    previous = _numeric(df, f"window_{previous_window}h_growth_rate", 0.0)
    return (current - previous).to_numpy()


def _split_indices(df: pd.DataFrame, labels: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    stratify = labels.to_numpy() if labels.nunique() > 1 and labels.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )
    return train_idx, test_idx


def _validation_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = train_df["high_future_growth"].astype(int)
    indices = np.arange(len(train_df))
    stratify = labels.to_numpy() if labels.nunique() > 1 and labels.value_counts().min() >= 2 else None
    fit_idx, valid_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=7,
        stratify=stratify,
    )
    return (
        train_df.loc[fit_idx].reset_index(drop=True),
        train_df.loc[valid_idx].reset_index(drop=True),
    )


def _ranking_metrics(y_future: np.ndarray, y_high: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y_high)) < 2 or len(np.unique(scores)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": round(float(roc_auc_score(y_high, scores)), 4),
        "average_precision": round(float(average_precision_score(y_high, scores)), 4),
    }


def _budget_metrics(
    y_future: np.ndarray,
    y_high: np.ndarray,
    scores: np.ndarray,
    budget_fraction: float,
) -> dict[str, Any]:
    k = max(1, int(np.ceil(len(y_future) * budget_fraction)))
    top_idx = np.argsort(scores)[::-1][:k]
    captured = float(y_future[top_idx].sum())
    total = float(y_future.sum())
    precision = float(y_high[top_idx].mean()) if len(top_idx) else 0.0
    base_rate = float(y_high.mean()) if len(y_high) else 0.0
    high_total = float(y_high.sum())
    recall = float(y_high[top_idx].sum() / high_total) if high_total else 0.0
    return {
        "budget_threads": int(k),
        "captured_future_growth": round(captured, 4),
        "capture_rate": round(captured / total, 4) if total > 0 else None,
        "precision_at_budget": round(precision, 4),
        "recall_at_budget": round(recall, 4),
        "lift_at_budget": round(precision / base_rate, 4) if base_rate > 0 else None,
    }


def _capture_rate(y_future: np.ndarray, scores: np.ndarray, budget_fraction: float) -> float:
    total = float(y_future.sum())
    if total <= 0:
        return 0.0
    k = max(1, int(np.ceil(len(y_future) * budget_fraction)))
    top_idx = np.argsort(scores)[::-1][:k]
    return float(y_future[top_idx].sum()) / total


def _build_summary(dataset_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"dataset": dataset_name, "status": "skipped", "reason": "no_valid_triage_rows"}
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(
            [
                "dataset",
                "cohort",
                "observation_window_h",
                "evaluation_slice",
                "variant",
                "feature_set",
                "model_family",
                "model_name",
                "training_objective",
                "budget_fraction",
            ],
            as_index=False,
        )
        .agg(
            test_threads=("test_threads", "mean"),
            capture_rate_mean=("capture_rate", "mean"),
            capture_rate_std=("capture_rate", "std"),
            precision_at_budget_mean=("precision_at_budget", "mean"),
            recall_at_budget_mean=("recall_at_budget", "mean"),
            lift_at_budget_mean=("lift_at_budget", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            average_precision_mean=("average_precision", "mean"),
        )
        .fillna({"capture_rate_std": 0.0})
    )
    summary_rows = grouped.to_dict(orient="records")
    significance_rows = _significance_rows(frame)
    deployable = grouped[
        (grouped["budget_fraction"] == PRIMARY_BUDGET)
        & (grouped["evaluation_slice"] == "all")
        & (grouped["variant"] != "oracle_upper_bound")
    ].copy()
    best_rows = []
    if not deployable.empty:
        best_index = deployable.groupby(["cohort", "observation_window_h"])["capture_rate_mean"].idxmax()
        best_rows = deployable.loc[best_index].sort_values(["cohort", "observation_window_h"]).to_dict(orient="records")
    return {
        "dataset": dataset_name,
        "status": "completed",
        "target": "future_growth_after_observation_window",
        "primary_budget_fraction": PRIMARY_BUDGET,
        "high_growth_quantile": HIGH_GROWTH_QUANTILE,
        "summary_rows": _round_records(summary_rows),
        "significance_rows": _round_records(significance_rows),
        "best_deployable_rows": _round_records(best_rows),
    }


def _significance_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, cohort, window, budget, evaluation_slice), setting_df in frame.groupby(
        ["dataset", "cohort", "observation_window_h", "budget_fraction", "evaluation_slice"]
    ):
        for primary_variant in SIGNIFICANCE_PRIMARY_VARIANTS:
            primary = setting_df[setting_df["variant"] == primary_variant]
            if primary.empty:
                continue
            primary_by_seed = primary.set_index("seed")["capture_rate"]
            candidate_baselines = sorted(
                variant
                for variant in setting_df["variant"].dropna().astype(str).unique()
                if variant not in {primary_variant, "oracle_upper_bound"}
            )
            for baseline in candidate_baselines:
                if baseline == primary_variant:
                    continue
                baseline_df = setting_df[setting_df["variant"] == baseline]
                if baseline_df.empty:
                    continue
                baseline_by_seed = baseline_df.set_index("seed")["capture_rate"]
                common_seeds = sorted(set(primary_by_seed.index) & set(baseline_by_seed.index))
                if len(common_seeds) < 2:
                    continue
                diff = (
                    primary_by_seed.loc[common_seeds].astype(float)
                    - baseline_by_seed.loc[common_seeds].astype(float)
                ).to_numpy()
                rows.append(
                    {
                        "dataset": dataset,
                        "cohort": cohort,
                        "observation_window_h": int(window),
                        "budget_fraction": float(budget),
                        "evaluation_slice": evaluation_slice,
                        "primary_variant": primary_variant,
                        "baseline_variant": baseline,
                        "n_seeds": len(common_seeds),
                        "primary_capture_mean": float(primary_by_seed.loc[common_seeds].mean()),
                        "baseline_capture_mean": float(baseline_by_seed.loc[common_seeds].mean()),
                        "mean_delta": float(np.mean(diff)),
                        "std_delta": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                        "delta_ci95_low": _mean_ci95(diff)[0],
                        "delta_ci95_high": _mean_ci95(diff)[1],
                        "p_value_sign_test": _paired_sign_test_pvalue(diff),
                        "p_value_paired_t": _paired_t_test_pvalue(diff),
                        "p_value_wilcoxon": _paired_wilcoxon_pvalue(diff),
                        "win_rate": float(np.mean(diff > 0)),
                    }
                )
    return rows


def _mean_ci95(diff: np.ndarray) -> tuple[float, float]:
    if len(diff) < 2:
        value = float(np.mean(diff)) if len(diff) else 0.0
        return value, value
    mean = float(np.mean(diff))
    stderr = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    half_width = 1.96 * stderr
    return mean - half_width, mean + half_width


def _paired_t_test_pvalue(diff: np.ndarray) -> float | None:
    if len(diff) < 2 or float(np.std(diff, ddof=1)) <= 1e-12:
        return 1.0 if np.all(np.abs(diff) <= 1e-12) else None
    try:
        from scipy import stats
    except Exception:
        return None
    result = stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
    return float(result.pvalue) if np.isfinite(result.pvalue) else None


def _paired_wilcoxon_pvalue(diff: np.ndarray) -> float | None:
    nonzero = diff[np.abs(diff) > 1e-12]
    if len(nonzero) < 2:
        return 1.0 if len(nonzero) == 0 else None
    try:
        from scipy import stats
    except Exception:
        return None
    try:
        result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return None
    return float(result.pvalue) if np.isfinite(result.pvalue) else None


def _paired_sign_test_pvalue(diff: np.ndarray) -> float | None:
    nonzero = diff[np.abs(diff) > 1e-12]
    n = int(len(nonzero))
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero > 0))
    tail = sum(_binom_pmf(n, k, 0.5) for k in range(0, min(wins, n - wins) + 1))
    return min(1.0, 2.0 * tail)


def _binom_pmf(n: int, k: int, p: float) -> float:
    from math import comb

    return comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def _round_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounded = []
    for row in rows:
        payload = {}
        for key, value in row.items():
            if isinstance(value, float):
                payload[key] = round(value, 4)
            else:
                payload[key] = value
        rounded.append(payload)
    return rounded


def _safe_scores(scores: Any, size: int) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    if array.shape[0] != size:
        array = np.zeros(size, dtype=float)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _numeric(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _observed_time(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    values = _numeric(df, column, window + 1.0)
    return values.where((values >= 0) & (values <= window), window + 1.0)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float).replace(0.0, np.nan)
    return (numerator.astype(float) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _truncate_text(text: str, max_chars: int) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def _text_baselines_enabled() -> bool:
    return os.getenv("RUMOR_SIM_ENABLE_NEURAL_TRIAGE", "1").strip().lower() not in {"0", "false", "no"}


def _triage_semantic_features_enabled() -> bool:
    return os.getenv("RUMOR_SIM_ENABLE_TRIAGE_SEMANTIC", "1").strip().lower() not in {"0", "false", "no"}


def _triage_semantic_model_name() -> str:
    return os.getenv("RUMOR_SIM_TRIAGE_SEMANTIC_MODEL", TRIAGE_SEMANTIC_MODEL_NAME)


def _triage_semantic_batch_size() -> int:
    return int(os.getenv("RUMOR_SIM_TRIAGE_SEMANTIC_BATCH_SIZE", str(TRIAGE_SEMANTIC_BATCH_SIZE)))


def _triage_semantic_max_chars() -> int:
    return int(os.getenv("RUMOR_SIM_TRIAGE_SEMANTIC_MAX_CHARS", str(TRIAGE_SEMANTIC_MAX_CHARS)))


def _text_model_name() -> str:
    return os.getenv("RUMOR_SIM_TRIAGE_TEXT_MODEL", TEXT_MODEL_NAME)


def _text_max_length() -> int:
    return int(os.getenv("RUMOR_SIM_TRIAGE_TEXT_MAX_LENGTH", str(TEXT_MAX_LENGTH)))


def _text_batch_size() -> int:
    return int(os.getenv("RUMOR_SIM_TRIAGE_TEXT_BATCH_SIZE", str(TEXT_BATCH_SIZE)))


def _text_epochs() -> int:
    return int(os.getenv("RUMOR_SIM_TRIAGE_TEXT_EPOCHS", str(TEXT_EPOCHS)))


def _variant_name(feature_set: str, model_kind: str) -> str:
    if model_kind == "lgbm_lambdarank":
        return f"{feature_set}_lgbm_lambdamart_ranker"
    if model_kind == "xgb_ranker":
        return f"{feature_set}_xgb_ltr_ranker"
    if model_kind == "catboost_ranker":
        return f"{feature_set}_catboost_ltr_ranker"
    if model_kind == "xlmr_regression":
        return "text_xlmr_regression_ranker"
    if model_kind == "xlmr_pairwise":
        return "text_xlmr_pairwise_ranker"
    return f"{feature_set}_{model_kind}_ranker"


def _model_family(model_kind: str) -> str:
    if model_kind in {"ridge", "lasso", "elasticnet"}:
        return "linear"
    if model_kind in {"rf", "hgb", "xgb", "lgbm", "catboost"}:
        return "tabular"
    if model_kind in LTR_MODEL_KINDS:
        return "learning_to_rank"
    return "model"


def _ltr_objective(model_kind: str) -> str:
    if model_kind == "lgbm_lambdarank":
        return "lambdamart_ndcg"
    if model_kind == "xgb_ranker":
        return "xgboost_rank_ndcg"
    if model_kind == "catboost_ranker":
        return "catboost_yetirank"
    if model_kind == "ranknet":
        return "pairwise_ranknet_bce"
    return "learning_to_rank"


def _text_objective(model_kind: str) -> str:
    if model_kind == "xlmr_regression":
        return "finetuned_xlmr_log_future_growth_regression"
    if model_kind == "xlmr_pairwise":
        return "finetuned_xlmr_pairwise_ranking"
    return "finetuned_text_ranking"


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    frame.to_csv(path, index=False)
