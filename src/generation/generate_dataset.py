"""Generate reproducible JSONL datasets from a JSON configuration.

每个样本生成后立即运行 validate_task；任何样本不合法则整体报错退出，
保证落盘数据 100% 程序可验证。

用法：python -m src.generation.generate_dataset --config configs/pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

from src.tasks.random_dag import generate_random_dag_task
from src.tasks.state_transition import generate_state_transition_task
from src.tasks.validators import validate_task


def generate_dataset(
    state_space_size: int,
    depths: list[int],
    n_per_cell: int,
    seed: int,
    avoid_cycle_within_depth: bool = False,
) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for depth in depths:
        for i in range(n_per_cell):
            task_id = f"st_s{state_space_size}_d{depth}_{seed}_{i:04d}"
            task = generate_state_transition_task(
                state_space_size=state_space_size,
                depth=depth,
                rng=rng,
                task_id=task_id,
                avoid_cycle_within_depth=avoid_cycle_within_depth,
            )
            problems = validate_task(task)
            if problems:
                raise AssertionError(f"generated invalid task {task_id}: {problems}")
            records.append(task.to_dict())
    rng.shuffle(records)
    return records


def generate_dag_dataset(
    work: int,
    depths: list[int],
    symbol_count: int,
    n_per_cell: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for depth in depths:
        for index in range(n_per_cell):
            task_id = f"dag_w{work}_d{depth}_{seed}_{index:04d}"
            task = generate_random_dag_task(
                work=work,
                depth=depth,
                symbol_count=symbol_count,
                rng=rng,
                task_id=task_id,
            )
            problems = validate_task(task)
            if problems:
                raise AssertionError(f"generated invalid task {task_id}: {problems}")
            records.append(task.to_dict())
    # Interleave experimental conditions to avoid warm-up or temporal drift
    # becoming a depth confound during model inference.
    rng.shuffle(records)
    return records


def generate_from_config(config: dict) -> list[dict]:
    family = config.get("task_family")
    common = {
        "depths": list(config["depths"]),
        "n_per_cell": int(config["n_per_cell"]),
        "seed": int(config["seed"]),
    }
    if family == "random_dag":
        return generate_dag_dataset(
            work=int(config["work"]),
            symbol_count=int(config["symbol_count"]),
            **common,
        )
    if family == "state_transition":
        return generate_dataset(
            state_space_size=int(config["state_space_size"]),
            avoid_cycle_within_depth=bool(
                config.get("avoid_cycle_within_depth", False)
            ),
            **common,
        )
    raise ValueError(f"unsupported task_family: {family!r}")


def write_dataset(records: list[dict], output_path: Path, config: dict) -> dict:
    """Atomically write data and a small reproducibility manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        for record in records:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            temporary.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(temporary_path, output_path)

    manifest = {
        "dataset": str(output_path),
        "sha256": digest.hexdigest(),
        "records": len(records),
        "config": config,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(manifest, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
    os.replace(temporary_path, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON experiment configuration")
    parser.add_argument("--out", help="输出 JSONL 路径（legacy state-transition mode）")
    parser.add_argument(
        "--state-space-size",
        type=int,
        default=16,
        help="|S|，所有条件保持恒定（默认 16）",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 24, 32],
        help="串行深度 D 列表（默认 pilot 配置）",
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=100,
        help="每个 (|S|, D) 格子的样本数（默认 100）",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--avoid-cycle-within-depth",
        action="store_true",
        help="要求轨迹在 D 步内不重复状态（需要 |S| > D）",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as file:
            config = json.load(file)
        records = generate_from_config(config)
        output_path = Path(config["out"])
    else:
        if not args.out:
            parser.error("either --config or --out is required")
        config = {
            "task_family": "state_transition",
            "state_space_size": args.state_space_size,
            "depths": args.depths,
            "n_per_cell": args.n_per_cell,
            "seed": args.seed,
            "avoid_cycle_within_depth": args.avoid_cycle_within_depth,
            "out": args.out,
        }
        records = generate_from_config(config)
        output_path = Path(args.out)

    manifest = write_dataset(records, output_path, config)
    print(f"wrote {len(records)} verified tasks to {output_path}")
    print(f"sha256={manifest['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
