"""数据集生成入口：按 (state_space_size, depth) 网格生成 JSONL 数据集。

每个样本生成后立即运行 validate_task；任何样本不合法则整体报错退出，
保证落盘数据 100% 程序可验证。

用法：
    python -m src.generation.generate_dataset \
        --out data/pilot/state_transition.jsonl \
        --state-space-size 16 \
        --depths 2 4 8 16 24 32 \
        --n-per-cell 100 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

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
                raise AssertionError(
                    f"generated invalid task {task_id}: {problems}"
                )
            records.append(task.to_dict())
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="输出 JSONL 路径")
    parser.add_argument("--state-space-size", type=int, default=16,
                        help="|S|，所有条件保持恒定（默认 16）")
    parser.add_argument("--depths", type=int, nargs="+",
                        default=[2, 4, 8, 16, 24, 32],
                        help="串行深度 D 列表（默认 pilot 配置）")
    parser.add_argument("--n-per-cell", type=int, default=100,
                        help="每个 (|S|, D) 格子的样本数（默认 100）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--avoid-cycle-within-depth", action="store_true",
                        help="要求轨迹在 D 步内不重复状态（需要 |S| > D）")
    args = parser.parse_args()

    records = generate_dataset(
        state_space_size=args.state_space_size,
        depths=args.depths,
        n_per_cell=args.n_per_cell,
        seed=args.seed,
        avoid_cycle_within_depth=args.avoid_cycle_within_depth,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} tasks to {args.out}")
    print(f"  state_space_size={args.state_space_size}, depths={args.depths}, "
          f"n_per_cell={args.n_per_cell}, seed={args.seed}")
    print("  all tasks passed validate_task (100% verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
