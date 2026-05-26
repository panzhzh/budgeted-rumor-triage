from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
import os
import traceback

from .audit import cross_dataset_report, dataset_quality_report
from .config import DATASET_NAMES, PipelineConfig, STAGES
from .datasets import parse_dataset
from .features import extract_thread_features
from .graph_analysis import compute_graph_metrics
from .io_utils import dump_json, ensure_dir, ensure_writable_dir
from .reporting import build_summary_tables
from .runtime import build_execution_plan, detect_cuda_runtime
from .status import log_message, stage_status_payload, utc_now_iso, write_status
from .text_models import build_text_model_features
from .triage import run_triage_analysis


def run_pipeline(
    config: PipelineConfig,
    *,
    dataset_names: tuple[str, ...] | None = None,
    stages: tuple[str, ...] | None = None,
    max_workers: int | None = None,
    reuse_text_features: bool = False,
) -> dict:
    selected_datasets = dataset_names or DATASET_NAMES
    selected_stages = stages or STAGES
    resolved_checkpoint_root = ensure_writable_dir(config.checkpoint_root)
    resolved_result_root = ensure_writable_dir(config.result_root)
    runtime_summary = {
        "checkpoint_root_requested": str(config.checkpoint_root),
        "checkpoint_root_resolved": str(resolved_checkpoint_root),
        "result_root_requested": str(config.result_root),
        "result_root_resolved": str(resolved_result_root),
    }

    effective_config = PipelineConfig(
        project_root=config.project_root,
        output_root=config.output_root,
        dataset_root=config.dataset_root,
        checkpoint_root=resolved_checkpoint_root,
        result_root=resolved_result_root,
        windows_hours=config.windows_hours,
        response_taxonomy=config.response_taxonomy,
        dataset_languages=config.dataset_languages,
        sentence_transformer_model=config.sentence_transformer_model,
        transformer_model=config.transformer_model,
        text_model_batch_size=config.text_model_batch_size,
        text_model_level=config.text_model_level,
        text_feature_root=config.text_feature_root,
        dataset_timezones=config.dataset_timezones,
        ignored_names=config.ignored_names,
        ignored_prefixes=config.ignored_prefixes,
    )

    plan = build_execution_plan(selected_datasets, max_workers=max_workers)
    runtime_summary["execution_plan"] = {
        "mode": plan.mode,
        "gpu_count": plan.gpu_info.count,
        "gpu_source": plan.gpu_info.source,
        "device_ids": list(plan.gpu_info.device_ids),
        "parallel_workers": plan.parallel_workers,
        "dataset_assignments": plan.dataset_assignments,
    }
    runtime_summary["cuda_runtime"] = detect_cuda_runtime()

    pipeline_status_path = effective_config.checkpoint_root / "status" / "pipeline_status.json"
    write_status(
        pipeline_status_path,
        {
            "status": "running",
            "stage": "pipeline",
            "datasets": list(selected_datasets),
            "execution_plan": runtime_summary["execution_plan"],
            "stages": list(selected_stages),
        },
    )
    pipeline_log_path = effective_config.checkpoint_root / "logs" / "pipeline.log"
    log_message(
        pipeline_log_path,
        f"pipeline started with mode={plan.mode}, workers={plan.parallel_workers}, gpu_count={plan.gpu_info.count}",
    )
    log_message(
        pipeline_log_path,
        "cuda_runtime="
        + str(runtime_summary["cuda_runtime"]),
    )

    results = {}
    start_time = perf_counter()
    if plan.parallel_workers <= 1:
        for dataset_name in selected_datasets:
            device = plan.dataset_assignments.get(dataset_name, "cpu")
            results[dataset_name] = _run_dataset_pipeline_with_env(
                effective_config,
                dataset_name,
                device,
                selected_stages,
                reuse_text_features,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=plan.parallel_workers,
            mp_context=get_context("spawn"),
        ) as executor:
            future_map = {}
            for dataset_name in selected_datasets:
                device = plan.dataset_assignments.get(dataset_name, "cpu")
                future = executor.submit(
                    _run_dataset_pipeline_with_env,
                    effective_config,
                    dataset_name,
                    device,
                    selected_stages,
                    reuse_text_features,
                )
                future_map[future] = dataset_name
            for future in as_completed(future_map):
                dataset_name = future_map[future]
                results[dataset_name] = future.result()

    report_paths = {
        dataset_name: Path(results[dataset_name]["audit"]["quality_report_path"])
        for dataset_name in selected_datasets
        if results[dataset_name]["audit"].get("quality_report_path")
    }
    cross_report_path = effective_config.result_stage_path("reports") / "cross_dataset_report.json"
    ensure_dir(cross_report_path.parent)
    cross = cross_dataset_report(report_paths, cross_report_path)
    table_manifest = build_summary_tables(effective_config.result_root)
    runtime_summary["wall_clock_seconds"] = round(perf_counter() - start_time, 3)
    log_message(
        pipeline_log_path,
        f"pipeline completed in {runtime_summary['wall_clock_seconds']}s; summary tables written to {effective_config.result_root / 'tables'}",
    )

    manifest = {
        "runtime": runtime_summary,
        "datasets": results,
        "cross_dataset_report_path": str(cross_report_path),
        "cross_dataset_report": cross["comparison"],
        "summary_tables": table_manifest,
    }
    dump_json(effective_config.result_root / "run_manifest.json", manifest)
    write_status(
        pipeline_status_path,
        {
            "status": "completed",
            "stage": "pipeline",
            "seconds": runtime_summary["wall_clock_seconds"],
            "result_root": str(effective_config.result_root),
            "checkpoint_root": str(effective_config.checkpoint_root),
        },
    )
    return manifest


def _run_dataset_pipeline_with_env(
    config: PipelineConfig,
    dataset_name: str,
    device: str,
    stages: tuple[str, ...],
    reuse_text_features: bool,
) -> dict:
    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return _run_dataset_pipeline(config, dataset_name, device, stages, reuse_text_features)


def _run_dataset_pipeline(
    config: PipelineConfig,
    dataset_name: str,
    device: str,
    stages: tuple[str, ...],
    reuse_text_features: bool,
) -> dict:
    dataset_path = config.dataset_path(dataset_name)
    normalized_dir = config.checkpoint_stage_path("normalized", dataset_name)
    logs_dir = config.checkpoint_stage_path("logs", dataset_name)
    status_dir = config.checkpoint_stage_path("status", dataset_name)
    features_dir = config.result_stage_path("features", dataset_name)
    graph_dir = config.result_stage_path("graphs", dataset_name)
    triage_dir = config.result_stage_path("triage", dataset_name)
    reports_dir = config.result_stage_path("reports", dataset_name)
    text_model_dir = config.result_stage_path("text_models", dataset_name)
    text_feature_store_dir = config.text_feature_path(dataset_name)

    for directory in (
        normalized_dir,
        logs_dir,
        status_dir,
        features_dir,
        graph_dir,
        triage_dir,
        reports_dir,
        text_model_dir,
    ):
        ensure_dir(directory)

    dataset_status_path = status_dir / "dataset_status.json"
    dataset_log_path = logs_dir / "dataset.log"
    write_status(
        dataset_status_path,
        {
            "dataset": dataset_name,
            "status": "running",
            "stage": "dataset",
            "device": device,
            "dataset_path": str(dataset_path),
        },
    )
    log_message(
        dataset_log_path,
        f"{dataset_name}: assigned device={device}, dataset_path={dataset_path}",
    )

    timings = {}
    canonical_jsonl = normalized_dir / "canonical.jsonl"
    canonical_csv = normalized_dir / "canonical.csv"
    parse_audit_json = reports_dir / "parse_audit.json"

    parse_audit = {}
    quality_report = {}
    feature_summary = {}
    text_model_summary = {}
    graph_summary = {}
    triage_summary = {}
    quality_report_path = reports_dir / "quality_report.json"
    feature_summary_path = features_dir / "feature_summary.json"
    graph_summary_path = graph_dir / "graph_summary.json"
    triage_summary_path = triage_dir / "triage_summary.json"
    text_model_summary_path = text_model_dir / "text_model_summary.json"
    normalize_seconds = audit_seconds = graph_seconds = feature_seconds = text_model_seconds = triage_seconds = 0.0

    try:
        if "normalize" in stages:
            parse_audit, normalize_seconds = _run_stage(
                dataset_name,
                "normalize",
                device,
                status_dir,
                dataset_log_path,
                lambda: parse_dataset(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    jsonl_path=canonical_jsonl,
                    csv_path=canonical_csv,
                    audit_path=parse_audit_json,
                    default_timezone=config.dataset_timezone(dataset_name),
                ),
                extra={
                    "canonical_jsonl": str(canonical_jsonl),
                    "canonical_csv": str(canonical_csv),
                },
            )
            timings["normalize_seconds"] = normalize_seconds
        elif parse_audit_json.exists():
            parse_audit = {}

        if "audit" in stages:
            quality_report, audit_seconds = _run_stage(
                dataset_name,
                "audit",
                device,
                status_dir,
                dataset_log_path,
                lambda: dataset_quality_report(canonical_csv, dataset_name),
                extra={"canonical_csv": str(canonical_csv)},
            )
            dump_json(quality_report_path, quality_report)
            timings["audit_seconds"] = audit_seconds
        elif quality_report_path.exists():
            import json
            quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))

        graph_csv = graph_dir / "graph_metrics.csv"
        if "graph" in stages:
            graph_summary, graph_seconds = _run_stage(
                dataset_name,
                "graph",
                device,
                status_dir,
                dataset_log_path,
                lambda: compute_graph_metrics(
                    canonical_csv,
                    dataset_name,
                    graph_csv,
                    graph_summary_path,
                ),
                extra={"graph_csv": str(graph_csv)},
            )
            timings["graph_seconds"] = graph_seconds

        features_csv = features_dir / "thread_features.csv"
        if "features" in stages:
            feature_summary, feature_seconds = _run_stage(
                dataset_name,
                "features",
                device,
                status_dir,
                dataset_log_path,
                lambda: extract_thread_features(
                    canonical_csv,
                    dataset_name,
                    features_csv,
                    feature_summary_path,
                    windows_hours=config.windows_hours,
                ),
                extra={"features_csv": str(features_csv)},
            )
            timings["feature_seconds"] = feature_seconds

        if "text_models" in stages:
            text_model_summary, text_model_seconds = _run_stage(
                dataset_name,
                "text_models",
                device,
                status_dir,
                dataset_log_path,
                lambda: build_text_model_features(
                    canonical_csv,
                    dataset_name,
                    text_model_dir,
                    feature_store_dir=text_feature_store_dir,
                    sentence_model_name=config.sentence_transformer_model,
                    transformer_model_name=config.transformer_model,
                    batch_size=config.text_model_batch_size,
                    text_model_level=config.text_model_level,
                    reuse_cache=reuse_text_features,
                ),
                extra={"text_model_dir": str(text_model_dir)},
            )
            timings["text_model_seconds"] = text_model_seconds

        if "triage" in stages:
            triage_summary, triage_seconds = _run_stage(
                dataset_name,
                "triage",
                device,
                status_dir,
                dataset_log_path,
                lambda: run_triage_analysis(
                    features_csv,
                    dataset_name,
                    triage_dir,
                    text_model_dir=text_model_dir,
                    canonical_csv_path=canonical_csv,
                    model_root=triage_dir / "models",
                ),
                extra={"triage_dir": str(triage_dir)},
            )
            timings["triage_seconds"] = triage_seconds

    except Exception as exc:
        log_message(dataset_log_path, f"{dataset_name}: failed with error={exc}")
        write_status(
            dataset_status_path,
            {
                "dataset": dataset_name,
                "status": "failed",
                "stage": "dataset",
                "device": device,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    write_status(
        dataset_status_path,
        {
            "dataset": dataset_name,
            "status": "completed",
            "stage": "dataset",
            "device": device,
            "timings": timings,
        },
    )
    log_message(dataset_log_path, f"{dataset_name}: completed with timings={timings}")

    return {
        "dataset_path": str(dataset_path),
        "device": device,
        "checkpoint_paths": {
            "canonical_jsonl": str(canonical_jsonl),
            "canonical_csv": str(canonical_csv),
            "status_dir": str(status_dir),
            "logs_dir": str(logs_dir),
        },
        "audit": {
            "parse": parse_audit,
            "parse_audit_path": str(parse_audit_json),
            "quality": quality_report,
            "quality_report_path": str(quality_report_path),
        },
        "features": {
            "summary": feature_summary,
            "features_csv": str(features_dir / "thread_features.csv"),
            "feature_summary_path": str(feature_summary_path),
        },
        "text_models": {
            "summary": text_model_summary,
            "text_model_dir": str(text_model_dir),
            "feature_store_dir": str(text_feature_store_dir),
            "text_model_summary_path": str(text_model_summary_path),
        },
        "graph": {
            "summary": graph_summary,
            "graph_csv": str(graph_dir / "graph_metrics.csv"),
            "graph_summary_path": str(graph_summary_path),
        },
        "triage": {
            "summary": triage_summary,
            "triage_dir": str(triage_dir),
            "triage_summary_path": str(triage_summary_path),
        },
        "timings": timings,
    }


def _run_stage(
    dataset_name: str,
    stage: str,
    device: str,
    status_dir: Path,
    log_path: Path,
    fn,
    extra: dict | None = None,
):
    stage_path = status_dir / f"{stage}_status.json"
    started_at = utc_now_iso()
    log_message(log_path, f"{dataset_name}:{stage} started on device={device}")
    write_status(
        stage_path,
        stage_status_payload(
            dataset_name=dataset_name,
            stage=stage,
            status="running",
            started_at=started_at,
            device=device,
            extra=extra,
        ),
    )
    started = perf_counter()
    try:
        result = fn()
    except Exception as exc:
        log_message(log_path, f"{dataset_name}:{stage} failed after {round(perf_counter() - started, 3)}s with error={exc}")
        write_status(
            stage_path,
            stage_status_payload(
                dataset_name=dataset_name,
                stage=stage,
                status="failed",
                started_at=started_at,
                finished_at=utc_now_iso(),
                seconds=perf_counter() - started,
                device=device,
                extra={
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    **(extra or {}),
                },
            ),
        )
        raise
    seconds = perf_counter() - started
    log_message(log_path, f"{dataset_name}:{stage} completed in {round(seconds, 3)}s")
    write_status(
        stage_path,
        stage_status_payload(
            dataset_name=dataset_name,
            stage=stage,
            status="completed",
            started_at=started_at,
            finished_at=utc_now_iso(),
            seconds=seconds,
            device=device,
            extra=extra,
        ),
    )
    return result, round(seconds, 3)
