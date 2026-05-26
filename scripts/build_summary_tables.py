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

from research.config import DEFAULT_RESULT_ROOT
from research.reporting import build_summary_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary tables and figures from completed EBRT results.")
    parser.add_argument(
        "--result-root",
        type=str,
        default=str(DEFAULT_RESULT_ROOT),
        help="Result root containing reports, features, triage outputs, and text-model summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_summary_tables(Path(args.result_root))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
