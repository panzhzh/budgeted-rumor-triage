from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .io_utils import dump_json, ensure_dir


def compute_graph_metrics(
    canonical_csv_path: Path,
    dataset_name: str,
    graph_csv_path: Path,
    summary_path: Path,
) -> dict:
    ensure_dir(graph_csv_path.parent)
    thread_rows = defaultdict(list)
    with canonical_csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            thread_rows[row["thread_id"]].append(row)
    rows = [
        _thread_graph_metrics(dataset_name, rows_for_thread)
        for _, rows_for_thread in sorted(thread_rows.items())
    ]

    with graph_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "dataset", "thread_id", "event_id", "thread_label", "node_count",
            "edge_count", "reaction_count", "max_depth", "max_breadth",
            "root_outdegree", "avg_branching_factor", "tree_edge_ratio",
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": dataset_name,
        "threads": len(rows),
        "avg_node_count": _safe_mean([row["node_count"] for row in rows]),
        "avg_reaction_count": _safe_mean([row["reaction_count"] for row in rows]),
        "avg_max_depth": _safe_mean([row["max_depth"] for row in rows]),
        "max_depth_overall": max((row["max_depth"] for row in rows), default=0),
        "avg_max_breadth": _safe_mean([row["max_breadth"] for row in rows]),
        "avg_root_outdegree": _safe_mean([row["root_outdegree"] for row in rows]),
        "avg_branching_factor": _safe_mean([row["avg_branching_factor"] for row in rows]),
    }
    dump_json(summary_path, summary)
    return summary


def _thread_graph_metrics(dataset_name: str, rows: list[dict]) -> dict:
    source = next((row for row in rows if row["post_type"] == "source"), rows[0])
    node_ids = {row["post_id"] for row in rows}
    root_id = source["post_id"]
    children: dict[str, list[str]] = {}
    depths: dict[str, int] = {root_id: 0}

    for row in rows:
        post_id = row["post_id"]
        if post_id not in children:
            children[post_id] = []

    edge_count = 0
    for row in rows:
        if row["post_type"] == "source":
            continue
        post_id = row["post_id"]
        parent_id = row.get("parent_id") or root_id
        if parent_id not in node_ids:
            parent_id = root_id
        children.setdefault(parent_id, []).append(post_id)
        edge_count += 1

    stack = [root_id]
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            if child in depths:
                continue
            depths[child] = depths[node] + 1
            stack.append(child)

    breadth: dict[int, int] = {}
    for depth in depths.values():
        breadth[depth] = breadth.get(depth, 0) + 1

    non_leaf_nodes = sum(1 for node, kids in children.items() if kids)
    avg_branching_factor = edge_count / non_leaf_nodes if non_leaf_nodes else 0.0
    node_count = len(rows)

    return {
        "dataset": dataset_name,
        "thread_id": source["thread_id"],
        "event_id": source["event_id"],
        "thread_label": source["thread_label"],
        "node_count": node_count,
        "edge_count": edge_count,
        "reaction_count": max(node_count - 1, 0),
        "max_depth": max(depths.values(), default=0),
        "max_breadth": max(breadth.values(), default=1 if node_count else 0),
        "root_outdegree": len(children.get(root_id, [])),
        "avg_branching_factor": round(avg_branching_factor, 4),
        "tree_edge_ratio": round(edge_count / max(node_count - 1, 1), 4) if node_count else 0.0,
    }


def _safe_mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 4)
