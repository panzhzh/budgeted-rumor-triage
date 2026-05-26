#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research.config import DATASET_NAMES, build_config
from research.datasets import parse_dataset
from research.io_utils import ensure_dir
from research.text_models import build_text_model_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute cached text features for early budgeted rumor triage datasets.")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--result-root", type=str, default=None)
    parser.add_argument("--text-feature-root", type=str, default=None)
    parser.add_argument("--text-model-level", choices=("light", "heavy"), default="light")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        result_root=args.result_root,
        text_feature_root=args.text_feature_root,
        text_model_level=args.text_model_level,
    )
    dataset_names = tuple(args.datasets) if args.datasets else DATASET_NAMES
    batch_size = args.batch_size or config.text_model_batch_size

    for dataset_name in dataset_names:
        dataset_path = config.dataset_path(dataset_name)
        normalized_dir = config.checkpoint_stage_path("normalized", dataset_name)
        ensure_dir(normalized_dir)
        canonical_jsonl = normalized_dir / "canonical.jsonl"
        canonical_csv = normalized_dir / "canonical.csv"
        parse_audit_json = config.result_stage_path("reports", dataset_name) / "parse_audit.json"
        if not canonical_csv.exists():
            print(f"[{dataset_name}] canonical not found, parsing source dataset first", flush=True)
            parse_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                jsonl_path=canonical_jsonl,
                csv_path=canonical_csv,
                audit_path=parse_audit_json,
                default_timezone=config.dataset_timezone(dataset_name),
            )
        print(f"[{dataset_name}] precomputing text features with level={config.text_model_level}", flush=True)
        build_text_model_features(
            canonical_csv,
            dataset_name,
            config.result_stage_path("text_models", dataset_name),
            feature_store_dir=config.text_feature_path(dataset_name),
            sentence_model_name=config.sentence_transformer_model,
            transformer_model_name=config.transformer_model,
            batch_size=batch_size,
            text_model_level=config.text_model_level,
            reuse_cache=args.reuse_cache,
        )
        print(f"[{dataset_name}] done", flush=True)


if __name__ == "__main__":
    main()
