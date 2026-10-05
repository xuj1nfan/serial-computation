"""Print accuracy and Wilson confidence intervals by work and depth."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - margin, center + margin


def summarize(path: Path) -> list[dict]:
    groups: dict[tuple[int | None, int], list[bool]] = defaultdict(list)
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok" or record["task_id"] in seen:
                continue
            seen.add(record["task_id"])
            groups[(record.get("work"), record["depth"])].append(record["correct"])

    rows: list[dict] = []
    for (work, depth), values in sorted(
        groups.items(), key=lambda item: ((item[0][0] or 0), item[0][1])
    ):
        correct = sum(values)
        low, high = wilson_interval(correct, len(values))
        rows.append(
            {
                "work": work,
                "depth": depth,
                "n": len(values),
                "correct": correct,
                "accuracy": correct / len(values),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    rows = summarize(args.path)
    print("work\tdepth\tn\tcorrect\taccuracy\t95% CI")
    for row in rows:
        work = "-" if row["work"] is None else str(row["work"])
        print(
            f"{work}\t{row['depth']}\t{row['n']}\t{row['correct']}\t"
            f"{row['accuracy']:.3f}\t"
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
