#!/usr/bin/env python3
"""Replay saved snapshot scores without fitting or loading a model.

Input JSON: thread_ids, source_hours, reaction_times (sorted relative-hour arrays),
scores (one [1h, 6h, 24h] score row per thread). Use --capacity in threads/hour.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research.replay import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--capacity", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    ids = [str(v) for v in data["thread_ids"]]
    source = np.asarray(data["source_hours"], float)
    times = [np.asarray(v, float) for v in data["reaction_times"]]
    scores = np.asarray(data["scores"], float)
    if not ids or len(set(ids)) != len(ids) or source.shape != (len(ids),) or len(times) != len(ids) or scores.shape != (len(ids), 3):
        raise ValueError("Thread arrays must be nonempty, aligned and uniquely identified")
    if not np.isfinite(source).all() or not np.isfinite(scores).all() or not math.isfinite(args.capacity) or args.capacity <= 0:
        raise ValueError("Finite inputs and positive capacity required")
    for t in times:
        if t.ndim != 1 or not np.isfinite(t).all() or (t < 0).any() or (np.diff(t) < 0).any():
            raise ValueError("Reaction times must be sorted, finite, nonnegative arrays")
    result = {policy: dict(zip(("summary", "decisions"), replay(ids, source, times, scores, args.capacity, policy)))
              for policy in ("combined_catboost_ranker", "early_volume", "fifo")}
    def finite(value):
        if isinstance(value, dict): return {k: finite(v) for k,v in value.items()}
        if isinstance(value, list): return [finite(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(finite(result), handle, indent=2, allow_nan=False)
        handle.write("\n")

if __name__ == "__main__":
    main()
