#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research.config import DATASET_NAMES, build_config
from research.runtime import detect_visible_gpus


def main() -> None:
    config = build_config()
    gpu_info = detect_visible_gpus()
    print("Early budgeted rumor triage dataset status")
    print(f"- dataset_root: {config.dataset_root}")
    print(f"- output_root: {config.output_root}")
    print(f"- checkpoint_root: {config.checkpoint_root}")
    print(f"- result_root: {config.result_root}")
    print(f"- visible_gpus: {gpu_info.count} ({gpu_info.source})")
    print(f"- gpu_device_ids: {list(gpu_info.device_ids)}")
    for name in DATASET_NAMES:
        path = config.dataset_path(name)
        print(f"- {name}: {path}")
        if path.exists():
            print("  status: present")
            if path.is_dir():
                children = [child for child in sorted(path.iterdir()) if not child.name.startswith(".")]
                print(f"  entries: {len(children)}")
                for child in children[:10]:
                    print(f"    - {child.name}")
        else:
            print("  status: missing")


if __name__ == "__main__":
    main()
