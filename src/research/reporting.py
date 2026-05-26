from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd

from .config import DATASET_LANGUAGES, DATASET_NAMES
from .io_utils import dump_json, ensure_dir


MAIN_TRIAGE_VARIANT = "combined_catboost_ranker"
MAIN_TABLE_VARIANTS = (
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
)
ALGORITHM_LABELS = {
    "random_budget": ("Random budget", "Random lower bound"),
    "early_volume": ("Early volume", "Simple heuristic"),
    "early_growth_rate": ("Early growth rate", "Simple heuristic"),
    "recent_acceleration": ("Recent acceleration", "Simple heuristic"),
    "text_xlmr_pairwise_ranker": ("XLM-R pairwise", "Text-only neural"),
    "combined_ridge_ranker": ("Combined Ridge", "Linear regression baseline"),
    "combined_rf_ranker": ("Combined Random Forest", "Tree regression baseline"),
    "response_xgb_ranker": ("Response XGB", "Response-only baseline"),
    "semantic_xgb_ranker": ("Semantic XGB", "Semantic-only baseline"),
    "cascade_catboost_ranker": ("Cascade CatBoost", "Cascade-only baseline"),
    "combined_xgb_ranker": ("Combined XGB", "Combined-feature ranker"),
    "combined_lgbm_lambdamart_ranker": ("Combined LambdaMART", "Direct learning-to-rank baseline"),
    "combined_xgb_ltr_ranker": ("Combined XGBRanker", "Direct learning-to-rank baseline"),
    "combined_catboost_ranker": ("Combined CatBoost", "Combined-feature ranker"),
    "oracle_upper_bound": ("Oracle upper bound", "Non-deployable upper bound"),
}
BUDGET_CURVE_VARIANTS = (
    "random_budget",
    "early_volume",
    "early_growth_rate",
    "text_xlmr_pairwise_ranker",
    "response_xgb_ranker",
    "combined_lgbm_lambdamart_ranker",
    "combined_xgb_ranker",
    "combined_catboost_ranker",
    "oracle_upper_bound",
)
COLD_START_TABLE_VARIANTS = (
    "early_volume",
    "text_xlmr_pairwise_ranker",
    "response_xgb_ranker",
    "semantic_xgb_ranker",
    "cascade_catboost_ranker",
    "combined_xgb_ranker",
    "combined_catboost_ranker",
)
SIGNAL_REGIME_VARIANTS = COLD_START_TABLE_VARIANTS
SIGNAL_LABELS = {
    "volume": "Volume",
    "text": "Text",
    "response": "Response",
    "semantic": "Semantic",
    "cascade": "Cascade",
    "combined": "Combined",
}
SIGNAL_ORDER = ("volume", "text", "response", "semantic", "cascade", "combined")
SIGNAL_COLORS = {
    "volume": "#E69F00",
    "text": "#56B4E9",
    "response": "#D55E00",
    "semantic": "#CC79A7",
    "cascade": "#009E73",
    "combined": "#0072B2",
}
RELIABILITY_COMPARISONS = (
    ("combined_catboost_ranker", "random_budget"),
    ("combined_catboost_ranker", "early_volume"),
    ("combined_catboost_ranker", "early_growth_rate"),
    ("combined_catboost_ranker", "recent_acceleration"),
    ("combined_catboost_ranker", "text_xlmr_pairwise_ranker"),
    ("combined_catboost_ranker", "combined_ridge_ranker"),
    ("combined_catboost_ranker", "combined_rf_ranker"),
    ("combined_catboost_ranker", "combined_lgbm_lambdamart_ranker"),
    ("combined_catboost_ranker", "response_xgb_ranker"),
    ("combined_xgb_ranker", "random_budget"),
    ("combined_xgb_ranker", "early_volume"),
    ("combined_xgb_ranker", "early_growth_rate"),
    ("combined_xgb_ranker", "recent_acceleration"),
    ("combined_xgb_ranker", "text_xlmr_pairwise_ranker"),
    ("combined_xgb_ranker", "combined_ridge_ranker"),
    ("combined_xgb_ranker", "combined_rf_ranker"),
    ("combined_xgb_ranker", "combined_lgbm_lambdamart_ranker"),
    ("combined_xgb_ranker", "response_xgb_ranker"),
    ("combined_xgb_ranker", "combined_catboost_ranker"),
)


def build_summary_tables(result_root: Path) -> dict:
    tables_dir = ensure_dir(result_root / "tables")
    analysis_dir = ensure_dir(result_root / "analysis")

    dataset_rows = []
    response_rows = []
    timing_rows = []
    graph_rows = []
    text_model_rows = []
    triage_rows = []
    triage_significance_rows = []

    for dataset_name in DATASET_NAMES:
        quality_path = result_root / "reports" / dataset_name / "quality_report.json"
        feature_path = result_root / "features" / dataset_name / "feature_summary.json"
        parse_path = result_root / "reports" / dataset_name / "parse_audit.json"
        if not quality_path.exists():
            continue

        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        feature = (
            json.loads(feature_path.read_text(encoding="utf-8"))
            if feature_path.exists()
            else {}
        )
        triage_path = result_root / "triage" / dataset_name / "triage_summary.json"
        graph_path = result_root / "graphs" / dataset_name / "graph_summary.json"
        parse = (
            json.loads(parse_path.read_text(encoding="utf-8"))
            if parse_path.exists()
            else {}
        )
        triage = (
            json.loads(triage_path.read_text(encoding="utf-8"))
            if triage_path.exists()
            else {}
        )
        text_model_path = result_root / "text_models" / dataset_name / "text_model_summary.json"
        graph = (
            json.loads(graph_path.read_text(encoding="utf-8"))
            if graph_path.exists()
            else {}
        )
        text_model = (
            json.loads(text_model_path.read_text(encoding="utf-8"))
            if text_model_path.exists()
            else {}
        )

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "threads": quality.get("thread_count"),
                "events": quality.get("event_count"),
                "source_posts": quality.get("source_posts"),
                "reaction_posts": quality.get("reaction_posts"),
                "timestamp_coverage_rate": quality.get("timestamp_coverage_rate"),
                "avg_reactions_per_thread": quality.get("avg_reactions_per_thread"),
                "avg_relative_hours": quality.get("avg_relative_hours"),
                "max_relative_hours": quality.get("max_relative_hours"),
                "feature_rows": feature.get("feature_rows"),
                "parse_timestamp_failures": parse.get("timestamp_parse_failures"),
            }
        )

        for response_type, count in sorted(
            (quality.get("stance_type_distribution") or {}).items()
        ):
            response_rows.append(
                {
                    "dataset": dataset_name,
                    "stance_type": response_type,
                    "count": count,
                    "share": round(count / max(quality.get("total_records", 1), 1), 4),
                }
            )

        timing_rows.append(
            {
                "dataset": dataset_name,
                "avg_relative_hours": quality.get("avg_relative_hours"),
                "max_relative_hours": quality.get("max_relative_hours"),
                "timestamp_coverage_rate": quality.get("timestamp_coverage_rate"),
            }
        )
        if graph:
            graph_rows.append(
                {
                    "dataset": dataset_name,
                    "threads": graph.get("threads"),
                    "avg_node_count": graph.get("avg_node_count"),
                    "avg_reaction_count": graph.get("avg_reaction_count"),
                    "avg_max_depth": graph.get("avg_max_depth"),
                    "max_depth_overall": graph.get("max_depth_overall"),
                    "avg_max_breadth": graph.get("avg_max_breadth"),
                    "avg_root_outdegree": graph.get("avg_root_outdegree"),
                    "avg_branching_factor": graph.get("avg_branching_factor"),
                }
            )
        if text_model:
            sentence_summary = text_model.get("sentence_transformer", {})
            correction_sentence_summary = text_model.get("correction_sentence_transformer", {})
            transformer_summary = text_model.get("zero_shot", {})
            thread_summary = text_model.get("thread_features", {})
            post_summary = text_model.get("post_features", {})
            text_model_rows.extend(
                [
                    {
                        "dataset": dataset_name,
                        "component": "sentence_transformer_source",
                        "status": sentence_summary.get("status"),
                        "model_name": sentence_summary.get("model_name"),
                        "row_count": _available_rows(sentence_summary, "rows_encoded"),
                    },
                    {
                        "dataset": dataset_name,
                        "component": "sentence_transformer_correction",
                        "status": correction_sentence_summary.get("status"),
                        "model_name": correction_sentence_summary.get("model_name"),
                        "row_count": _available_rows(correction_sentence_summary, "rows_encoded"),
                    },
                    {
                        "dataset": dataset_name,
                        "component": "zero_shot",
                        "status": transformer_summary.get("status"),
                        "model_name": transformer_summary.get("model_name"),
                        "row_count": _available_rows(transformer_summary, "rows_scored"),
                    },
                    {
                        "dataset": dataset_name,
                        "component": "post_text_features",
                        "status": "completed" if post_summary.get("rows") else "skipped",
                        "model_name": text_model.get("text_model_level"),
                        "row_count": post_summary.get("rows"),
                    },
                    {
                        "dataset": dataset_name,
                        "component": "thread_text_features",
                        "status": thread_summary.get("status"),
                        "model_name": "aggregated",
                        "row_count": thread_summary.get("rows"),
                    },
                ]
            )
        if triage.get("status") == "completed":
            for row in triage.get("summary_rows") or []:
                triage_rows.append(
                    {
                        "dataset": row.get("dataset"),
                        "cohort": row.get("cohort"),
                        "observation_window_h": row.get("observation_window_h"),
                        "evaluation_slice": row.get("evaluation_slice", "all"),
                        "variant": row.get("variant"),
                        "feature_set": row.get("feature_set"),
                        "model_family": row.get("model_family"),
                        "model_name": row.get("model_name"),
                        "training_objective": row.get("training_objective"),
                        "budget_fraction": row.get("budget_fraction"),
                        "test_threads": row.get("test_threads"),
                        "capture_rate_mean": row.get("capture_rate_mean"),
                        "capture_rate_std": row.get("capture_rate_std"),
                        "precision_at_budget_mean": row.get("precision_at_budget_mean"),
                        "recall_at_budget_mean": row.get("recall_at_budget_mean"),
                        "lift_at_budget_mean": row.get("lift_at_budget_mean"),
                        "roc_auc_mean": row.get("roc_auc_mean"),
                        "average_precision_mean": row.get("average_precision_mean"),
                    }
                )
            for row in triage.get("significance_rows") or []:
                triage_significance_rows.append(row)

    _purge_legacy_reporting_outputs(tables_dir)
    _write_csv(
        tables_dir / "raw_dataset_overview.csv",
        dataset_rows,
    )
    _write_csv(
        tables_dir / "raw_graph_summary.csv",
        graph_rows,
    )
    _write_csv(
        tables_dir / "raw_text_model_summary.csv",
        text_model_rows,
    )
    _write_csv(
        tables_dir / "raw_representative_triage_summary.csv",
        triage_rows,
    )
    _write_csv(
        tables_dir / "raw_representative_triage_significance.csv",
        triage_significance_rows,
    )
    main_table_rows = _main_table_rows(triage_rows)
    dataset_signal_rows = _dataset_signal_availability_rows(result_root, dataset_rows, graph_rows)
    budget_curve_rows = _budget_capture_curve_rows(triage_rows)
    dataset_window_rows = _dataset_window_actionability_rows(triage_rows)
    signal_mechanism_rows = _signal_mechanism_rows(triage_rows)
    cold_start_rows = _cold_start_rows(triage_rows)
    reliability_rows = _statistical_reliability_rows(triage_rows)
    signal_regime_heatmap_rows = _signal_regime_heatmap_rows(triage_rows)
    signal_regime_transition_rows = _signal_regime_transition_rows(triage_rows)
    figure_manifest = _build_final_figures(
        result_root,
        budget_curve_rows,
        signal_regime_heatmap_rows,
        signal_regime_transition_rows,
    )
    _write_csv(
        tables_dir / "dataset_signal_availability.csv",
        dataset_signal_rows,
    )
    _write_csv(
        tables_dir / "main_table_results.csv",
        main_table_rows,
    )
    _write_csv(
        tables_dir / "budget_capture_bars.csv",
        budget_curve_rows,
    )
    _write_csv(
        tables_dir / "dataset_window_actionability.csv",
        dataset_window_rows,
    )
    _write_csv(
        tables_dir / "signal_mechanism_deltas.csv",
        signal_mechanism_rows,
    )
    _write_csv(
        tables_dir / "cold_start_results.csv",
        cold_start_rows,
    )
    _write_csv(
        tables_dir / "statistical_reliability.csv",
        reliability_rows,
    )
    _write_csv(
        tables_dir / "signal_regime_heatmap.csv",
        signal_regime_heatmap_rows,
    )
    _write_csv(
        tables_dir / "signal_regime_transitions.csv",
        signal_regime_transition_rows,
    )

    markdown_summary = _build_markdown_summary(dataset_rows, triage_rows)
    (analysis_dir / "analysis_summary.md").write_text(markdown_summary, encoding="utf-8")
    _write_text_if_lines(
        analysis_dir / "main_table_findings.md",
        _build_main_table_findings(main_table_rows),
    )

    manifest = {
        "dataset_signal_availability_csv": str(tables_dir / "dataset_signal_availability.csv"),
        "main_table_results_csv": str(tables_dir / "main_table_results.csv"),
        "budget_capture_bars_csv": str(tables_dir / "budget_capture_bars.csv"),
        "dataset_window_actionability_csv": str(tables_dir / "dataset_window_actionability.csv"),
        "signal_mechanism_deltas_csv": str(tables_dir / "signal_mechanism_deltas.csv"),
        "cold_start_results_csv": str(tables_dir / "cold_start_results.csv"),
        "statistical_reliability_csv": str(tables_dir / "statistical_reliability.csv"),
        "signal_regime_heatmap_csv": str(tables_dir / "signal_regime_heatmap.csv"),
        "signal_regime_transitions_csv": str(tables_dir / "signal_regime_transitions.csv"),
        "raw_representative_triage_summary_csv": str(tables_dir / "raw_representative_triage_summary.csv"),
        "raw_representative_triage_significance_csv": str(tables_dir / "raw_representative_triage_significance.csv"),
        "analysis_summary_md": str(analysis_dir / "analysis_summary.md"),
        "main_table_findings_md": str(analysis_dir / "main_table_findings.md"),
        "figure_manifest": figure_manifest,
    }
    dump_json(analysis_dir / "table_manifest.json", manifest)
    return manifest


def _write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        if path.exists():
            path.unlink()
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text_if_lines(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    if not text.strip():
        if path.exists():
            path.unlink()
        return
    path.write_text(text, encoding="utf-8")


def _purge_legacy_reporting_outputs(tables_dir: Path) -> None:
    legacy_names = (
        "dataset_overview.csv",
        "policy_comparison.csv",
        "response_distribution.csv",
        "timing_summary.csv",
        "model_summary.csv",
        "graph_summary.csv",
        "text_model_summary.csv",
        "triage_summary.csv",
        "triage_significance.csv",
        "triage_budget_curve.csv",
        "budget_capture_curves.csv",
        "triage_ablation.csv",
        "triage_cold_start.csv",
        "strong_baseline_summary.csv",
        "strong_baseline_best_by_setting.csv",
    )
    for name in legacy_names:
        path = tables_dir / name
        if path.exists():
            path.unlink()


def _dataset_signal_availability_rows(
    result_root: Path,
    dataset_rows: list[dict],
    graph_rows: list[dict],
) -> list[dict]:
    graph_by_dataset = {row.get("dataset"): row for row in graph_rows}
    rows: list[dict] = []
    for dataset in DATASET_NAMES:
        dataset_row = next((row for row in dataset_rows if row.get("dataset") == dataset), None)
        if dataset_row is None:
            continue
        graph_row = graph_by_dataset.get(dataset, {})
        feature_path = result_root / "features" / dataset / "thread_features.csv"
        availability = _early_signal_availability(feature_path)
        rows.append(
            {
                "dataset": dataset,
                "language": DATASET_LANGUAGES.get(dataset),
                "threads": dataset_row.get("threads"),
                "reactions": dataset_row.get("reaction_posts"),
                "mean_reactions": dataset_row.get("avg_reactions_per_thread"),
                "timestamp_coverage_rate": dataset_row.get("timestamp_coverage_rate"),
                "avg_max_depth": graph_row.get("avg_max_depth"),
                "max_depth_overall": graph_row.get("max_depth_overall"),
                "avg_root_outdegree": graph_row.get("avg_root_outdegree"),
                "response_available_1h": availability.get("response_available_1h"),
                "median_reactions_1h": availability.get("median_reactions_1h"),
                "response_available_6h": availability.get("response_available_6h"),
                "median_reactions_6h": availability.get("median_reactions_6h"),
                "response_available_24h": availability.get("response_available_24h"),
                "median_reactions_24h": availability.get("median_reactions_24h"),
            }
        )
    return rows


def _early_signal_availability(feature_path: Path) -> dict[str, float | None]:
    if not feature_path.exists():
        return {}
    df = pd.read_csv(feature_path)
    output: dict[str, float | None] = {}
    for window in (1, 6, 24):
        column = f"window_{window}h_reactions"
        if column not in df.columns or df.empty:
            output[f"response_available_{window}h"] = None
            output[f"median_reactions_{window}h"] = None
            continue
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        output[f"response_available_{window}h"] = round(float((values > 0).mean()), 4)
        output[f"median_reactions_{window}h"] = round(float(values.median()), 4)
    return output


def _main_table_rows(triage_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for variant in MAIN_TABLE_VARIANTS:
        all_rows = [
            row
            for row in triage_rows
            if row.get("variant") == variant and row.get("evaluation_slice", "all") == "all"
        ]
        cold_rows = [
            row
            for row in triage_rows
            if row.get("variant") == variant and row.get("evaluation_slice", "all") == "cold_start"
        ]
        if not all_rows:
            continue
        algorithm, role = ALGORITHM_LABELS[variant]
        rows.append(
            {
                "algorithm": algorithm,
                "variant": variant,
                "role": role,
                "capture_at_5": _metric_mean(all_rows, "capture_rate_mean", budget=0.05),
                "capture_at_10": _metric_mean(all_rows, "capture_rate_mean", budget=0.10),
                "capture_at_20": _metric_mean(all_rows, "capture_rate_mean", budget=0.20),
                "lift_at_10": _metric_mean(all_rows, "lift_at_budget_mean", budget=0.10),
                "roc_auc": _metric_mean(all_rows, "roc_auc_mean", budget=0.10),
                "average_precision": _metric_mean(all_rows, "average_precision_mean", budget=0.10),
                "cold_capture_at_10": _metric_mean(cold_rows, "capture_rate_mean", budget=0.10),
            }
        )
    return rows


def _metric_mean(rows: list[dict], key: str, *, budget: float) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and abs(float(row.get("budget_fraction") or 0.0) - budget) < 1e-9
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _budget_capture_curve_rows(triage_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for variant in BUDGET_CURVE_VARIANTS:
        for budget in (0.05, 0.10, 0.20):
            setting_rows = [
                row
                for row in triage_rows
                if row.get("variant") == variant
                and row.get("evaluation_slice", "all") == "all"
                and abs(float(row.get("budget_fraction") or 0.0) - budget) < 1e-9
            ]
            if not setting_rows:
                continue
            algorithm, role = ALGORITHM_LABELS[variant]
            rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "role": role,
                    "budget_fraction": budget,
                    "capture_rate": _mean(setting_rows, "capture_rate_mean"),
                    "capture_rate_std": _std(setting_rows, "capture_rate_mean"),
                    "mean_seed_std": _mean(setting_rows, "capture_rate_std"),
                    "lift_at_budget": _mean(setting_rows, "lift_at_budget_mean"),
                    "roc_auc": _mean(setting_rows, "roc_auc_mean"),
                    "average_precision": _mean(setting_rows, "average_precision_mean"),
                }
            )
    return rows


def _dataset_window_actionability_rows(triage_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASET_NAMES:
        for window in (1, 6, 24):
            primary_rows = _select_rows(
                triage_rows,
                variant=MAIN_TRIAGE_VARIANT,
                dataset=dataset,
                window=window,
                budget=0.10,
                evaluation_slice="all",
            )
            random_rows = _select_rows(
                triage_rows,
                variant="random_budget",
                dataset=dataset,
                window=window,
                budget=0.10,
                evaluation_slice="all",
            )
            early_rows = _select_rows(
                triage_rows,
                variant="early_volume",
                dataset=dataset,
                window=window,
                budget=0.10,
                evaluation_slice="all",
            )
            if not primary_rows:
                continue
            primary_capture = _mean(primary_rows, "capture_rate_mean")
            random_capture = _mean(random_rows, "capture_rate_mean")
            early_capture = _mean(early_rows, "capture_rate_mean")
            rows.append(
                {
                    "dataset": dataset,
                    "observation_window_h": window,
                    "combined_catboost_capture_at_10": primary_capture,
                    "random_capture_at_10": random_capture,
                    "early_volume_capture_at_10": early_capture,
                    "delta_vs_random": _rounded_delta(primary_capture, random_capture),
                    "delta_vs_early_volume": _rounded_delta(primary_capture, early_capture),
                    "lift_at_10": _mean(primary_rows, "lift_at_budget_mean"),
                    "average_precision": _mean(primary_rows, "average_precision_mean"),
                }
            )
    return rows


def _signal_mechanism_rows(triage_rows: list[dict]) -> list[dict]:
    comparisons = (
        (
            "Semantic XGB vs Response XGB",
            "semantic_xgb_ranker",
            "response_xgb_ranker",
            "semantic structured risk beyond response-only XGB",
        ),
        (
            "Combined XGB vs Response XGB",
            "combined_xgb_ranker",
            "response_xgb_ranker",
            "combined structured signals beyond response-only XGB",
        ),
        (
            "Combined XGB vs Semantic XGB",
            "combined_xgb_ranker",
            "semantic_xgb_ranker",
            "response/cascade maturity beyond semantic-only XGB",
        ),
        (
            "Combined CatBoost vs Cascade CatBoost",
            "combined_catboost_ranker",
            "cascade_catboost_ranker",
            "full combined representation beyond cascade-only CatBoost",
        ),
    )
    rows: list[dict] = []
    for label, primary_variant, baseline_variant, interpretation in comparisons:
        primary_all = _select_rows(triage_rows, variant=primary_variant, budget=0.10, evaluation_slice="all")
        baseline_all = _select_rows(triage_rows, variant=baseline_variant, budget=0.10, evaluation_slice="all")
        primary_cold = _select_rows(triage_rows, variant=primary_variant, budget=0.10, evaluation_slice="cold_start")
        baseline_cold = _select_rows(triage_rows, variant=baseline_variant, budget=0.10, evaluation_slice="cold_start")
        primary_all_c = _mean(primary_all, "capture_rate_mean")
        baseline_all_c = _mean(baseline_all, "capture_rate_mean")
        primary_cold_c = _mean(primary_cold, "capture_rate_mean")
        baseline_cold_c = _mean(baseline_cold, "capture_rate_mean")
        if primary_all_c is None or baseline_all_c is None:
            continue
        rows.append(
            {
                "comparison": label,
                "primary_variant": primary_variant,
                "baseline_variant": baseline_variant,
                "primary_capture_at_10": primary_all_c,
                "baseline_capture_at_10": baseline_all_c,
                "delta_capture_at_10": _rounded_delta(primary_all_c, baseline_all_c),
                "primary_cold_capture_at_10": primary_cold_c,
                "baseline_cold_capture_at_10": baseline_cold_c,
                "delta_cold_capture_at_10": _rounded_delta(primary_cold_c, baseline_cold_c),
                "interpretation": interpretation,
            }
        )
    return rows


def _cold_start_rows(triage_rows: list[dict]) -> list[dict]:
    mechanisms = {
        "early_volume": "early response signal collapses when response evidence is sparse",
        "text_xlmr_pairwise_ranker": "source text remains available before responses mature",
        "response_xgb_ranker": "response dynamics weaken under sparse early evidence",
        "semantic_xgb_ranker": "semantic features retain prior risk information",
        "cascade_catboost_ranker": "cascade features need observed propagation structure",
        "combined_xgb_ranker": "combined structured signal remains more stable than single groups",
        "combined_catboost_ranker": "combined-feature trade-off under low-response evidence",
    }
    rows: list[dict] = []
    for variant in COLD_START_TABLE_VARIANTS:
        overall = _select_rows(triage_rows, variant=variant, budget=0.10, evaluation_slice="all")
        cold = _select_rows(triage_rows, variant=variant, budget=0.10, evaluation_slice="cold_start")
        if not overall or not cold:
            continue
        overall_capture = _mean(overall, "capture_rate_mean")
        cold_capture = _mean(cold, "capture_rate_mean")
        algorithm, role = ALGORITHM_LABELS[variant]
        rows.append(
            {
                "algorithm": algorithm,
                "variant": variant,
                "role": role,
                "overall_capture_at_10": overall_capture,
                "cold_capture_at_10": cold_capture,
                "cold_overall_ratio": _safe_ratio(cold_capture, overall_capture),
                "mechanism": mechanisms[variant],
            }
        )
    return rows


def _statistical_reliability_rows(triage_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for primary_variant, baseline_variant in RELIABILITY_COMPARISONS:
        paired = _paired_setting_deltas(
            triage_rows,
            primary_variant=primary_variant,
            baseline_variant=baseline_variant,
        )
        if not paired:
            continue
        deltas = [item["delta_capture_at_10"] for item in paired]
        ci_low, ci_high = _mean_ci95(deltas)
        primary_label = ALGORITHM_LABELS[primary_variant][0]
        baseline_label = ALGORITHM_LABELS[baseline_variant][0]
        rows.append(
            {
                "comparison": f"{primary_label} vs {baseline_label}",
                "primary_variant": primary_variant,
                "baseline_variant": baseline_variant,
                "n_settings": len(deltas),
                "mean_delta_capture_at_10": round(sum(deltas) / len(deltas), 4),
                "mean_delta_ci95_low": ci_low,
                "mean_delta_ci95_high": ci_high,
                "median_delta_capture_at_10": _median(deltas),
                "wins": sum(1 for value in deltas if value > 1e-12),
                "losses": sum(1 for value in deltas if value < -1e-12),
                "ties": sum(1 for value in deltas if abs(value) <= 1e-12),
                "wilcoxon_p": _wilcoxon_pvalue(deltas),
            }
        )
    _attach_holm_adjusted_pvalues(rows, "wilcoxon_p", "holm_adjusted_p")
    return rows


def _paired_setting_deltas(
    triage_rows: list[dict],
    *,
    primary_variant: str,
    baseline_variant: str,
) -> list[dict]:
    primary_by_setting = _setting_capture_map(
        _select_rows(
            triage_rows,
            variant=primary_variant,
            budget=0.10,
            evaluation_slice="all",
        )
    )
    baseline_by_setting = _setting_capture_map(
        _select_rows(
            triage_rows,
            variant=baseline_variant,
            budget=0.10,
            evaluation_slice="all",
        )
    )
    rows: list[dict] = []
    for setting in sorted(set(primary_by_setting) & set(baseline_by_setting)):
        primary = primary_by_setting[setting]
        baseline = baseline_by_setting[setting]
        rows.append(
            {
                "dataset": setting[0],
                "cohort": setting[1],
                "observation_window_h": setting[2],
                "primary_capture_at_10": primary,
                "baseline_capture_at_10": baseline,
                "delta_capture_at_10": round(primary - baseline, 4),
            }
        )
    return rows


def _setting_capture_map(rows: list[dict]) -> dict[tuple[str, str, int], float]:
    mapping: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        if row.get("capture_rate_mean") is None:
            continue
        key = (
            str(row.get("dataset")),
            str(row.get("cohort")),
            int(row.get("observation_window_h")),
        )
        mapping.setdefault(key, []).append(float(row["capture_rate_mean"]))
    return {key: round(sum(values) / len(values), 4) for key, values in mapping.items()}


def _select_rows(
    rows: list[dict],
    *,
    variant: str,
    budget: float,
    evaluation_slice: str,
    dataset: str | None = None,
    window: int | None = None,
) -> list[dict]:
    selected = []
    for row in rows:
        if row.get("variant") != variant:
            continue
        if row.get("evaluation_slice", "all") != evaluation_slice:
            continue
        if abs(float(row.get("budget_fraction") or 0.0) - budget) > 1e-9:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if window is not None and int(row.get("observation_window_h")) != int(window):
            continue
        selected.append(row)
    return selected


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _std(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return round(math.sqrt(variance), 4)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return round(sorted_values[middle], 4)
    return round((sorted_values[middle - 1] + sorted_values[middle]) / 2.0, 4)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return round(numerator / denominator, 4)


def _rounded_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 4)


def _mean_ci95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    if len(values) == 1:
        value = round(float(values[0]), 4)
        return (value, value)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    try:
        from scipy.stats import t

        multiplier = float(t.ppf(0.975, len(values) - 1))
    except Exception:
        multiplier = 1.96
    return (round(mean - multiplier * standard_error, 4), round(mean + multiplier * standard_error, 4))


def _wilcoxon_pvalue(deltas: list[float]) -> float | None:
    nonzero = [value for value in deltas if abs(value) > 1e-12]
    if len(nonzero) < 2:
        return 1.0 if not nonzero else None
    try:
        from scipy.stats import wilcoxon

        return round(float(wilcoxon(nonzero, alternative="greater").pvalue), 6)
    except Exception:
        return None


def _attach_holm_adjusted_pvalues(rows: list[dict], source_key: str, target_key: str) -> None:
    indexed = [
        (index, float(row[source_key]))
        for index, row in enumerate(rows)
        if row.get(source_key) is not None
    ]
    if not indexed:
        return
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running_max = 0.0
    for rank, (index, p_value) in enumerate(indexed, start=1):
        adjusted = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, adjusted)
        rows[index][target_key] = round(running_max, 6)
    for row in rows:
        row.setdefault(target_key, None)


def _signal_regime_heatmap_rows(triage_rows: list[dict]) -> list[dict]:
    setting_order = _signal_setting_order(triage_rows)
    rows: list[dict] = []
    for setting_index, setting in enumerate(setting_order):
        dataset, cohort, window, evaluation_slice = setting
        random_capture = _setting_variant_capture(
            triage_rows,
            setting=setting,
            variant="random_budget",
        )
        oracle_capture = _setting_variant_capture(
            triage_rows,
            setting=setting,
            variant="oracle_upper_bound",
        )
        if random_capture is None or oracle_capture is None:
            continue
        denominator = max(oracle_capture - random_capture, 1e-12)
        captures: dict[str, float] = {}
        for variant in SIGNAL_REGIME_VARIANTS:
            capture = _setting_variant_capture(triage_rows, setting=setting, variant=variant)
            if capture is not None:
                captures[variant] = capture
        if not captures:
            continue
        best_variant = max(
            SIGNAL_REGIME_VARIANTS,
            key=lambda item: captures.get(item, float("-inf")),
        )
        for variant in SIGNAL_REGIME_VARIANTS:
            capture = captures.get(variant)
            if capture is None:
                continue
            algorithm, role = ALGORITHM_LABELS[variant]
            source_row = _setting_variant_row(triage_rows, setting=setting, variant=variant)
            feature_set = str(source_row.get("feature_set")) if source_row else ""
            normalized = (capture - random_capture) / denominator
            rows.append(
                {
                    "setting_index": setting_index,
                    "setting_id": _format_signal_setting(setting),
                    "dataset": dataset,
                    "cohort": cohort,
                    "observation_window_h": window,
                    "evaluation_slice": evaluation_slice,
                    "algorithm": algorithm,
                    "variant": variant,
                    "feature_set": feature_set,
                    "signal": SIGNAL_LABELS.get(feature_set, feature_set),
                    "capture_at_10": round(capture, 4),
                    "random_capture_at_10": round(random_capture, 4),
                    "oracle_capture_at_10": round(oracle_capture, 4),
                    "oracle_normalized_capture": round(normalized, 4),
                    "is_setting_best": variant == best_variant,
                }
            )
    return rows


def _signal_regime_transition_rows(triage_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    base_settings = sorted(
        {
            (dataset, cohort, window)
            for dataset, cohort, window, _slice in _signal_setting_order(triage_rows)
        },
        key=lambda item: (
            DATASET_NAMES.index(item[0]) if item[0] in DATASET_NAMES else 999,
            ("all_threads", "rumor_only").index(item[1])
            if item[1] in ("all_threads", "rumor_only")
            else 999,
            int(item[2]),
        ),
    )
    for dataset, cohort, window in base_settings:
        all_setting = (dataset, cohort, window, "all")
        cold_setting = (dataset, cohort, window, "cold_start")
        all_best = _best_signal_variant(triage_rows, all_setting)
        cold_best = _best_signal_variant(triage_rows, cold_setting)
        if all_best is None or cold_best is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "cohort": cohort,
                "observation_window_h": window,
                "full_signal": SIGNAL_LABELS.get(all_best["feature_set"], all_best["feature_set"]),
                "full_feature_set": all_best["feature_set"],
                "full_algorithm": ALGORITHM_LABELS[all_best["variant"]][0],
                "full_variant": all_best["variant"],
                "full_capture_at_10": round(float(all_best["capture"]), 4),
                "cold_signal": SIGNAL_LABELS.get(cold_best["feature_set"], cold_best["feature_set"]),
                "cold_feature_set": cold_best["feature_set"],
                "cold_algorithm": ALGORITHM_LABELS[cold_best["variant"]][0],
                "cold_variant": cold_best["variant"],
                "cold_capture_at_10": round(float(cold_best["capture"]), 4),
                "signal_changed": all_best["feature_set"] != cold_best["feature_set"],
            }
        )
    return rows


def _signal_setting_order(triage_rows: list[dict]) -> list[tuple[str, str, int, str]]:
    available = {
        (
            str(row.get("dataset")),
            str(row.get("cohort")),
            int(row.get("observation_window_h")),
            str(row.get("evaluation_slice", "all")),
        )
        for row in triage_rows
        if row.get("variant") in SIGNAL_REGIME_VARIANTS
        and abs(float(row.get("budget_fraction") or 0.0) - 0.10) < 1e-9
        and row.get("capture_rate_mean") is not None
    }
    order: list[tuple[str, str, int, str]] = []
    for dataset in DATASET_NAMES:
        for cohort in ("all_threads", "rumor_only"):
            for window in (1, 6, 24):
                for evaluation_slice in ("all", "cold_start"):
                    setting = (dataset, cohort, window, evaluation_slice)
                    if setting in available:
                        order.append(setting)
    return order


def _setting_variant_row(
    triage_rows: list[dict],
    *,
    setting: tuple[str, str, int, str],
    variant: str,
) -> dict | None:
    dataset, cohort, window, evaluation_slice = setting
    matches = [
        row
        for row in triage_rows
        if row.get("dataset") == dataset
        and row.get("cohort") == cohort
        and int(row.get("observation_window_h")) == int(window)
        and row.get("evaluation_slice", "all") == evaluation_slice
        and row.get("variant") == variant
        and abs(float(row.get("budget_fraction") or 0.0) - 0.10) < 1e-9
        and row.get("capture_rate_mean") is not None
    ]
    return matches[0] if matches else None


def _setting_variant_capture(
    triage_rows: list[dict],
    *,
    setting: tuple[str, str, int, str],
    variant: str,
) -> float | None:
    row = _setting_variant_row(triage_rows, setting=setting, variant=variant)
    if row is None:
        return None
    return float(row["capture_rate_mean"])


def _best_signal_variant(
    triage_rows: list[dict],
    setting: tuple[str, str, int, str],
) -> dict | None:
    best: dict | None = None
    for variant in SIGNAL_REGIME_VARIANTS:
        row = _setting_variant_row(triage_rows, setting=setting, variant=variant)
        if row is None:
            continue
        capture = float(row["capture_rate_mean"])
        candidate = {
            "variant": variant,
            "feature_set": str(row.get("feature_set")),
            "capture": capture,
        }
        if best is None or capture > float(best["capture"]):
            best = candidate
    return best


def _format_signal_setting(setting: tuple[str, str, int, str]) -> str:
    dataset, cohort, window, evaluation_slice = setting
    cohort_label = "all" if cohort == "all_threads" else "rumor"
    slice_label = "cold" if evaluation_slice == "cold_start" else "all"
    return f"{dataset}|{cohort_label}|{window}h|{slice_label}"


def _build_final_figures(
    result_root: Path,
    budget_curve_rows: list[dict],
    signal_heatmap_rows: list[dict],
    signal_transition_rows: list[dict],
) -> dict:
    manifest: dict[str, str] = {}
    figure_dir = ensure_dir(result_root / "figures")
    if budget_curve_rows:
        manifest.update(_plot_budget_capture_figure(figure_dir, budget_curve_rows))
    if signal_heatmap_rows:
        manifest.update(_plot_signal_heatmap_figure(figure_dir, signal_heatmap_rows))
    if signal_transition_rows:
        manifest.update(_plot_signal_alluvial_figure(figure_dir, signal_transition_rows))
    return manifest


def _plot_budget_capture_figure(
    figure_dir: Path,
    budget_curve_rows: list[dict],
) -> dict:
    manifest: dict[str, str] = {}
    output_path = figure_dir / "figure_budget_capture_bars.png"
    pdf_path = figure_dir / "figure_budget_capture_bars.pdf"
    svg_path = figure_dir / "figure_budget_capture_bars.svg"
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_plot_style(plt)
        order = list(BUDGET_CURVE_VARIANTS)
        okabe_ito = [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#E69F00",  # orange
            "#CC79A7",  # reddish purple
            "#009E73",  # green
            "#56B4E9",  # sky blue
            "#999999",  # gray
        ]
        budgets = [0.05, 0.10, 0.20]
        budget_labels = ["5%", "10%", "20%"]
        hatches = ["", "/", ".", "\\", "x", "-", "+"]
        x = np.arange(len(budgets)) * 1.45
        width = 0.13
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        for index, variant in enumerate(order):
            rows = [row for row in budget_curve_rows if row.get("variant") == variant]
            if not rows:
                continue
            by_budget = {round(float(row["budget_fraction"]), 2): row for row in rows}
            means = [float(by_budget[budget]["capture_rate"]) for budget in budgets]
            errors = [float(by_budget[budget].get("capture_rate_std") or 0.0) for budget in budgets]
            offset = (index - (len(order) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                color=okabe_ito[index % len(okabe_ito)],
                edgecolor="#4a4a4a",
                linewidth=0.35,
                hatch=hatches[index % len(hatches)],
                yerr=errors,
                capsize=2.4,
                error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
                label=ALGORITHM_LABELS[variant][0],
            )
        ax.set_xlabel("Review budget")
        ax.set_ylabel("Future-growth capture", labelpad=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(budget_labels)
        group_half_width = ((len(order) - 1) / 2.0 + 0.5) * width
        ax.set_xlim(
            x[0] - group_half_width,
            x[-1] + group_half_width,
        )
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.28, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            columnspacing=1.35,
            handlelength=1.6,
        )
        fig.subplots_adjust(left=0.010, right=0.9995, bottom=0.135, top=0.805)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        _remove_legacy_figure_files(
            [figure_dir],
            "figure_budget_capture_curves",
        )
        manifest["budget_capture_bars_png"] = str(output_path)
        manifest["budget_capture_bars_pdf"] = str(pdf_path)
        manifest["budget_capture_bars_svg"] = str(svg_path)
    except Exception as exc:
        manifest["budget_capture_bars_error"] = str(exc)
    return manifest


def _plot_signal_heatmap_figure(
    figure_dir: Path,
    rows: list[dict],
) -> dict:
    manifest: dict[str, str] = {}
    output_path = figure_dir / "figure_signal_regime_heatmap.png"
    pdf_path = figure_dir / "figure_signal_regime_heatmap.pdf"
    svg_path = figure_dir / "figure_signal_regime_heatmap.svg"
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np

        _set_plot_style(plt)
        setting_ids = []
        for row in sorted(rows, key=lambda item: int(item["setting_index"])):
            setting_id = str(row["setting_id"])
            if setting_id not in setting_ids:
                setting_ids.append(setting_id)
        variants = [variant for variant in SIGNAL_REGIME_VARIANTS if any(row["variant"] == variant for row in rows)]
        matrix = np.full((len(variants), len(setting_ids)), np.nan)
        best_points: list[tuple[int, int]] = []
        row_lookup = {variant: index for index, variant in enumerate(variants)}
        column_lookup = {setting_id: index for index, setting_id in enumerate(setting_ids)}
        setting_meta: dict[str, dict] = {}
        for row in rows:
            r = row_lookup[str(row["variant"])]
            c = column_lookup[str(row["setting_id"])]
            value = float(row["oracle_normalized_capture"])
            matrix[r, c] = min(max(value, 0.0), 1.0)
            setting_meta[str(row["setting_id"])] = row
            if str(row.get("is_setting_best")).lower() == "true":
                best_points.append((r, c))

        fig, ax = plt.subplots(figsize=(13.8, 5.7))
        capture_cmap = LinearSegmentedColormap.from_list(
            "signal_capture_map",
            [
                (0.00, "#F7FBFD"),
                (0.18, "#D8EEF6"),
                (0.42, "#9BD2E5"),
                (0.66, "#4FAFC4"),
                (0.84, "#168C9B"),
                (1.00, "#005A9C"),
            ],
        )
        capture_cmap.set_bad("#F2F2F2")
        image = ax.imshow(matrix, aspect="auto", cmap=capture_cmap, vmin=0.0, vmax=1.0)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([ALGORITHM_LABELS[variant][0] for variant in variants])
        labels = []
        for setting_id in setting_ids:
            meta = setting_meta[setting_id]
            slice_label = "SR" if meta["evaluation_slice"] == "cold_start" else "All"
            labels.append(f"{int(meta['observation_window_h'])}h\n{slice_label}")
        ax.set_xticks(range(len(setting_ids)))
        ax.set_xticklabels(labels, rotation=0, fontsize=10.5)
        ax.set_xlabel("")
        ax.set_ylabel("Deployable ranker")
        ax.set_xticks(np.arange(-0.5, len(setting_ids), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_index, column_index in best_points:
            ax.scatter(
                column_index,
                row_index,
                s=30,
                marker="o",
                facecolors="none",
                edgecolors="#111111",
                linewidths=0.9,
            )
        for boundary in range(6, len(setting_ids), 6):
            ax.axvline(boundary - 0.5, color="#111111", linewidth=0.85)
        for start in range(0, len(setting_ids), 6):
            end = min(start + 5, len(setting_ids) - 1)
            center = (start + end) / 2.0
            meta = setting_meta[setting_ids[start]]
            cohort_label = "all" if meta["cohort"] == "all_threads" else "rumor"
            ax.text(
                center,
                -0.76,
                f"{str(meta['dataset'])} {cohort_label}",
                ha="center",
                va="bottom",
                fontsize=13,
                clip_on=False,
            )
        colorbar = fig.colorbar(image, ax=ax, pad=0.012, shrink=0.88)
        colorbar.set_label("Oracle-normalized capture")
        ax.text(
            1.0,
            -0.14,
            "Open circles mark the best deployable ranker within each setting.",
            ha="right",
            va="center",
            fontsize=10.5,
            transform=ax.transAxes,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        manifest["signal_regime_heatmap_png"] = str(output_path)
        manifest["signal_regime_heatmap_pdf"] = str(pdf_path)
        manifest["signal_regime_heatmap_svg"] = str(svg_path)
    except Exception as exc:
        manifest["signal_regime_heatmap_error"] = str(exc)
    return manifest


def _plot_signal_alluvial_figure(
    figure_dir: Path,
    rows: list[dict],
) -> dict:
    manifest: dict[str, str] = {}
    output_path = figure_dir / "figure_signal_migration_sankey.png"
    pdf_path = figure_dir / "figure_signal_migration_sankey.pdf"
    svg_path = figure_dir / "figure_signal_migration_sankey.svg"
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import PathPatch, Rectangle

        _set_plot_style(plt)
        transitions: dict[tuple[str, str], int] = {}
        left_counts: dict[str, int] = {}
        right_counts: dict[str, int] = {}
        for row in rows:
            source = str(row["full_feature_set"])
            target = str(row["cold_feature_set"])
            transitions[(source, target)] = transitions.get((source, target), 0) + 1
            left_counts[source] = left_counts.get(source, 0) + 1
            right_counts[target] = right_counts.get(target, 0) + 1
        left_signals = [signal for signal in SIGNAL_ORDER if left_counts.get(signal)]
        right_signals = [signal for signal in SIGNAL_ORDER if right_counts.get(signal)]
        total = max(sum(left_counts.values()), 1)
        fig, ax = plt.subplots(figsize=(9.8, 5.2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        left_extents, unit = _alluvial_node_extents(left_counts, left_signals)
        right_extents, _ = _alluvial_node_extents(right_counts, right_signals, unit=unit)
        left_cursor = {signal: top for signal, (_bottom, top) in left_extents.items()}
        right_cursor = {signal: top for signal, (_bottom, top) in right_extents.items()}
        x_left = 0.17
        x_right = 0.80
        node_width = 0.035
        for source in left_signals:
            for target in right_signals:
                value = transitions.get((source, target), 0)
                if value <= 0:
                    continue
                height = value * unit
                y0_top = left_cursor[source]
                y0_bottom = y0_top - height
                y1_top = right_cursor[target]
                y1_bottom = y1_top - height
                left_cursor[source] = y0_bottom
                right_cursor[target] = y1_bottom
                _draw_alluvial_band(
                    ax,
                    x_left + node_width,
                    x_right,
                    y0_bottom,
                    y0_top,
                    y1_bottom,
                    y1_top,
                    SIGNAL_COLORS.get(source, "#777777"),
                )
        for signal in left_signals:
            bottom, top = left_extents[signal]
            ax.add_patch(
                Rectangle(
                    (x_left, bottom),
                    node_width,
                    top - bottom,
                    facecolor=SIGNAL_COLORS.get(signal, "#777777"),
                    edgecolor="#111111",
                    linewidth=0.6,
                )
            )
            ax.text(
                x_left - 0.018,
                (bottom + top) / 2,
                f"{SIGNAL_LABELS.get(signal, signal)} {left_counts[signal]}",
                ha="right",
                va="center",
                fontsize=14,
            )
        for signal in right_signals:
            bottom, top = right_extents[signal]
            ax.add_patch(
                Rectangle(
                    (x_right, bottom),
                    node_width,
                    top - bottom,
                    facecolor=SIGNAL_COLORS.get(signal, "#777777"),
                    edgecolor="#111111",
                    linewidth=0.6,
                )
            )
            ax.text(
                x_right + node_width + 0.018,
                (bottom + top) / 2,
                f"{SIGNAL_LABELS.get(signal, signal)} {right_counts[signal]}",
                ha="left",
                va="center",
                fontsize=14,
            )
        ax.text(
            x_left + node_width / 2,
            0.94,
            "Full cohort winner",
            ha="center",
            va="bottom",
            fontsize=14,
        )
        ax.text(
            x_right + node_width / 2,
            0.94,
            "Sparse-response winner",
            ha="center",
            va="bottom",
            fontsize=14,
        )
        ax.text(
            0.5,
            0.045,
            f"Band width represents matched dataset, cohort, and window settings. Total settings = {total}.",
            ha="center",
            va="center",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        _remove_legacy_figure_files(
            [figure_dir],
            "figure_signal_regime_alluvial",
        )
        manifest["signal_migration_sankey_png"] = str(output_path)
        manifest["signal_migration_sankey_pdf"] = str(pdf_path)
        manifest["signal_migration_sankey_svg"] = str(svg_path)
    except Exception as exc:
        manifest["signal_migration_sankey_error"] = str(exc)
    return manifest


def _remove_legacy_figure_files(directories: list[Path], stem: str) -> None:
    for directory in directories:
        for suffix in (".png", ".pdf", ".svg"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def _set_plot_style(plt) -> None:
    plt.rcParams["hatch.linewidth"] = 0.45
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "font.family": "serif",
            "font.serif": [
                "Times",
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "pdf.use14corefonts": True,
            "pdf.fonttype": 42,
            "ps.useafm": True,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _alluvial_node_extents(
    counts: dict[str, int],
    signals: list[str],
    *,
    unit: float | None = None,
) -> tuple[dict[str, tuple[float, float]], float]:
    bottom_margin = 0.13
    top_margin = 0.88
    gap = 0.018
    total = max(sum(counts.get(signal, 0) for signal in signals), 1)
    if unit is None:
        usable = top_margin - bottom_margin - gap * max(len(signals) - 1, 0)
        unit = usable / total
    extents: dict[str, tuple[float, float]] = {}
    cursor = top_margin
    for signal in signals:
        height = counts.get(signal, 0) * unit
        extents[signal] = (cursor - height, cursor)
        cursor -= height + gap
    return extents, unit


def _draw_alluvial_band(
    ax,
    x0: float,
    x1: float,
    y0_bottom: float,
    y0_top: float,
    y1_bottom: float,
    y1_top: float,
    color: str,
) -> None:
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    control = 0.18
    vertices = [
        (x0, y0_top),
        (x0 + control, y0_top),
        (x1 - control, y1_top),
        (x1, y1_top),
        (x1, y1_bottom),
        (x1 - control, y1_bottom),
        (x0 + control, y0_bottom),
        (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=0.48,
        )
    )


def _build_main_table_findings(main_table_rows: list[dict]) -> str:
    if not main_table_rows:
        return ""
    lines = ["# Main Table Findings", ""]
    for row in main_table_rows:
        lines.append(
            f"- {row['algorithm']}: C@10={row.get('capture_at_10')}, "
            f"L@10={row.get('lift_at_10')}, cold C@10={row.get('cold_capture_at_10')}"
        )
    lines.append("")
    return "\n".join(lines)


def _available_rows(summary: dict, primary_key: str) -> int | None:
    if not summary:
        return None
    primary = summary.get(primary_key) or 0
    reused = summary.get("rows_reused") or 0
    return int(primary) + int(reused)


def _build_markdown_summary(
    dataset_rows: list[dict],
    triage_rows: list[dict],
) -> str:
    return _build_final_markdown_summary(dataset_rows, triage_rows)

def _build_final_markdown_summary(
    dataset_rows: list[dict],
    triage_rows: list[dict],
) -> str:
    lines = ["# Analysis Summary", "", "## Dataset Overview", ""]
    for row in dataset_rows:
        lines.append(
            f"- {row['dataset']}: {row['threads']} threads, {row['reaction_posts']} reactions, "
            f"timestamp coverage {row['timestamp_coverage_rate']}, avg reactions/thread {row['avg_reactions_per_thread']}"
        )
    if triage_rows:
        lines.extend(["", "## Representative Budgeted Triage", ""])
        for row in _main_table_rows(triage_rows):
            lines.append(
                f"- {row['algorithm']}: C@10={row.get('capture_at_10')}, "
                f"C@20={row.get('capture_at_20')}, cold C@10={row.get('cold_capture_at_10')}"
            )
        lines.extend(["", "## Dataset-Window Actionability", ""])
        for row in _dataset_window_actionability_rows(triage_rows):
            lines.append(
                f"- {row['dataset']} / {row['observation_window_h']}h: "
                f"Combined CatBoost C@10={row.get('combined_catboost_capture_at_10')}, "
                f"delta vs early volume={row.get('delta_vs_early_volume')}"
            )
    lines.append("")
    return "\n".join(lines)
