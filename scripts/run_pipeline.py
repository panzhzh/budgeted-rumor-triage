#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research.config import DATASET_NAMES, STAGES, build_config
from research.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the early budgeted rumor triage pipeline.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument("--output-root", type=str, default=None, help="Override unified output root path.")
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Override checkpoint root path.")
    parser.add_argument("--result-root", type=str, default=None, help="Override result root path.")
    parser.add_argument("--text-feature-root", type=str, default=None, help="Override cached text feature root.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of datasets to run. Choices: CHECKED CSDC-Rumor PHEME",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Optional subset of stages to run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional override for max dataset workers.",
    )
    parser.add_argument(
        "--reuse-text-features",
        action="store_true",
        help="Reuse cached post/thread text features from the feature store.",
    )
    parser.add_argument(
        "--text-model-level",
        choices=("light", "heavy"),
        default=None,
        help="Light uses prototype similarity only; heavy additionally computes multilingual zero-shot scores.",
    )
    parser.add_argument(
        "--skip-text-models",
        action="store_true",
        help="Skip text model stage entirely.",
    )
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
    print("Starting early budgeted rumor triage pipeline", flush=True)
    print(f"dataset_root={config.dataset_root}", flush=True)
    print(f"output_root={config.output_root}", flush=True)
    print(f"checkpoint_root={config.checkpoint_root}", flush=True)
    print(f"result_root={config.result_root}", flush=True)
    dataset_names = tuple(args.datasets) if args.datasets else DATASET_NAMES
    stages = tuple(args.stages) if args.stages else STAGES
    if args.skip_text_models:
        stages = tuple(stage for stage in stages if stage != "text_models")
    manifest = run_pipeline(
        config,
        dataset_names=dataset_names,
        stages=stages,
        max_workers=args.max_workers,
        reuse_text_features=args.reuse_text_features,
    )
    print(json.dumps(manifest["runtime"], ensure_ascii=False, indent=2))
    print(f"Manifest written to: {manifest['runtime']['result_root_resolved']}/run_manifest.json")


if __name__ == "__main__":
    main()
