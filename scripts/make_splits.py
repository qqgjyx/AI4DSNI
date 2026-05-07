#!/usr/bin/env python
"""Genus-aware greedy stratified split for DSNI.

Implements the algorithm described in the paper (Section 3.3): genera are sorted
by size in descending order, then each genus is greedily assigned to the bucket
(train/val/test) that minimizes a weighted deviation from target proportions in
both bucket size and per-task class distribution. This preserves stratification
over the four cultivation labels (media, temperature, pH, oxygen) while keeping
all sequences from one genus inside a single split.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TASKS = ["media", "temperature", "ph", "oxygen"]


def greedy_stratified_split(
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    tasks: List[str] = DEFAULT_TASKS,
    label_weight: float = 0.1,
    group_col: str = "genus",
    id_col: str = "strain_id",
) -> Dict[str, List[str]]:
    """Assign each genus to train/val/test minimizing a deviation score.

    Args:
        metadata: DataFrame with columns ``id_col``, ``group_col`` and ``tasks``.
        train_ratio, val_ratio, test_ratio: target proportions; must sum to 1.
        tasks: label columns over which to enforce stratification.
        label_weight: weight on per-class deviation relative to size deviation.
        group_col: grouping column (default genus).
        id_col: per-row identifier copied into the output split lists.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` of strain ids.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    targets = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    tasks = [t for t in tasks if t in metadata.columns]

    total_n = len(metadata)
    total_label_counts = {
        t: metadata[t].value_counts(dropna=False).to_dict() for t in tasks
    }

    groups = metadata.groupby(group_col, sort=False)
    genera_sorted = sorted(groups.groups, key=lambda g: -len(groups.get_group(g)))

    bucket_size: Dict[str, int] = {b: 0 for b in targets}
    bucket_label: Dict[str, Dict[str, Dict[object, int]]] = {
        b: {t: defaultdict(int) for t in tasks} for b in targets
    }
    assignments: Dict[str, List[str]] = {b: [] for b in targets}

    def deviation(bucket: str, rows: pd.DataFrame) -> float:
        size_after = bucket_size[bucket] + len(rows)
        score = abs(size_after - targets[bucket] * total_n) / max(total_n, 1)
        for t in tasks:
            counts = rows[t].value_counts(dropna=False)
            for cls, total_c in total_label_counts[t].items():
                cur = bucket_label[bucket][t][cls] + int(counts.get(cls, 0))
                target_c = targets[bucket] * total_c
                score += label_weight * abs(cur - target_c) / max(total_c, 1)
        return score

    for genus in genera_sorted:
        rows = groups.get_group(genus)
        best_b = min(targets, key=lambda b: deviation(b, rows))
        bucket_size[best_b] += len(rows)
        for t in tasks:
            for cls, cnt in rows[t].value_counts(dropna=False).items():
                bucket_label[best_b][t][cls] += int(cnt)
        assignments[best_b].extend(rows[id_col].astype(str).tolist())

    return assignments


def _summarize(metadata: pd.DataFrame, assignments: Dict[str, List[str]]) -> Dict:
    by_id = metadata.set_index("strain_id")
    summary = {}
    for split, ids in assignments.items():
        sub = by_id.loc[ids]
        summary[split] = {
            "n_sequences": len(sub),
            "n_genera": sub["genus"].nunique() if "genus" in sub.columns else None,
            "label_distribution": {
                t: sub[t].value_counts(dropna=False).to_dict()
                for t in DEFAULT_TASKS if t in sub.columns
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--label_weight", type=float, default=0.1)
    parser.add_argument(
        "--tasks", nargs="*", default=DEFAULT_TASKS,
        help="Label columns to stratify over.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metadata = pd.read_csv(args.metadata)

    assignments = greedy_stratified_split(
        metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        tasks=args.tasks,
        label_weight=args.label_weight,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in assignments.items():
        (args.out_dir / f"{split}.txt").write_text("\n".join(ids) + "\n")

    summary = _summarize(metadata, assignments)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    for split, info in summary.items():
        logger.info(
            "%s: %d sequences across %s genera",
            split, info["n_sequences"], info["n_genera"],
        )


if __name__ == "__main__":
    main()
