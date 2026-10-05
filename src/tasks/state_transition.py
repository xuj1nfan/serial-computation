"""Task Family A: Random State Transition.

生成随机有限状态系统 f: S -> S，给定初始状态 s0，要求计算 s_D = f^{(D)}(s0)。
每个样本携带完整 ground-truth 轨迹 [s0, s1, ..., s_D]，供 executor 独立复算
与 checkpoint intervention 使用。
"""

from __future__ import annotations

import random
import string
from dataclasses import asdict, dataclass, field

TASK_FAMILY = "state_transition"

# 状态名池：单个大写字母，保证 tokenization 干净、答案空间离散。
STATE_NAME_POOL = list(string.ascii_uppercase)  # A..Z, 最多 26 个状态


@dataclass
class StateTransitionTask:
    task_id: str
    task_family: str
    states: list[str]  # 系统中的全部状态名（prompt 中展示顺序已随机化）
    transitions: dict[str, str]  # 完整映射表 f: S -> S
    start_state: str  # s0
    depth: int  # D
    trajectory: list[str]  # ground truth: [s0, s1, ..., s_D]
    final_state: str  # s_D == trajectory[-1]
    prompt: str  # Direct Answer prompt（见论文方案第 8 节）
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StateTransitionTask:
        return cls(**d)


def execute_transition(
    transitions: dict[str, str], start_state: str, steps: int
) -> list[str]:
    """Exact ground-truth executor：逐步迭代 f，返回完整轨迹 [s0, ..., s_steps]。"""
    if start_state not in transitions:
        raise KeyError(f"start state {start_state!r} not in transition table")
    trajectory = [start_state]
    current = start_state
    for _ in range(steps):
        current = transitions[current]
        trajectory.append(current)
    return trajectory


def build_direct_prompt(
    states: list[str], transitions: dict[str, str], start_state: str, depth: int
) -> str:
    """Direct Answer prompt，严格遵循论文方案第 8 节，不暗示任何算法步骤。"""
    lines = [
        "You are given a state-transition system.",
        "",
        "Transitions:",
    ]
    for s in states:
        lines.append(f"{s} -> {transitions[s]}")
    lines += [
        "",
        f"Starting state: {start_state}",
        f"Apply the transition rule exactly {depth} times.",
        "",
        "Return only the final state.",
    ]
    return "\n".join(lines)


def build_checkpoint_prompt(task: StateTransitionTask, checkpoint_step: int) -> str:
    """Checkpoint intervention prompt（第 9 节）：提供真实中间状态 s_k，
    让模型从 s_k 继续剩余 D-k 步。"""
    if not 1 <= checkpoint_step < task.depth:
        raise ValueError(
            f"checkpoint_step must be in [1, {task.depth - 1}], got {checkpoint_step}"
        )
    s_k = task.trajectory[checkpoint_step]
    remaining = task.depth - checkpoint_step
    lines = [
        "You are given a state-transition system.",
        "",
        "Transitions:",
    ]
    for s in task.states:
        lines.append(f"{s} -> {task.transitions[s]}")
    lines += [
        "",
        f"The process has already been executed for {checkpoint_step} steps.",
        "",
        f"The correct state after step {checkpoint_step} is: {s_k}",
        "",
        f"Continue from {s_k} for the remaining {remaining} steps.",
        "",
        "Return only the final state.",
    ]
    return "\n".join(lines)


def generate_state_transition_task(
    state_space_size: int,
    depth: int,
    rng: random.Random,
    task_id: str,
    avoid_cycle_within_depth: bool = False,
    max_resample: int = 100,
) -> StateTransitionTask:
    """生成一个随机状态转移样本。

    - state_space_size: |S|，在同一实验条件下保持恒定（Answer Space Control）。
    - depth: 串行深度 D。
    - avoid_cycle_within_depth: 若为 True，直接构造一条 D 步内不重复的轨迹
      （要求 |S| > D）。默认 False，允许周期；这类样本只适合作为辅助任务，
      因为周期可能缩短实际计算路径。
    """
    if not 2 <= state_space_size <= len(STATE_NAME_POOL):
        raise ValueError(f"state_space_size must be in [2, {len(STATE_NAME_POOL)}]")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if avoid_cycle_within_depth and state_space_size <= depth:
        raise ValueError(
            "avoid_cycle_within_depth requires state_space_size > depth "
            f"(got |S|={state_space_size}, D={depth})"
        )

    # ``max_resample`` remains in the signature for backward compatibility. The
    # cycle-free path is now constructed directly instead of relying on a very
    # low-probability rejection sampler.
    del max_resample
    states = rng.sample(STATE_NAME_POOL, state_space_size)
    if avoid_cycle_within_depth:
        trajectory = rng.sample(states, depth + 1)
        transitions = {trajectory[i]: trajectory[i + 1] for i in range(depth)}
        for state in states:
            transitions.setdefault(state, rng.choice(states))
        start_state = trajectory[0]
    else:
        transitions = {s: rng.choice(states) for s in states}
        start_state = rng.choice(states)
        trajectory = execute_transition(transitions, start_state, depth)

    prompt = build_direct_prompt(states, transitions, start_state, depth)
    first_repeat_step = next(
        (
            index
            for index, state in enumerate(trajectory)
            if state in trajectory[:index]
        ),
        None,
    )
    return StateTransitionTask(
        task_id=task_id,
        task_family=TASK_FAMILY,
        states=states,
        transitions=transitions,
        start_state=start_state,
        depth=depth,
        trajectory=trajectory,
        final_state=trajectory[-1],
        prompt=prompt,
        metadata={
            "state_space_size": state_space_size,
            "prompt_format": "direct_answer",
            "cycle_free_within_depth": first_repeat_step is None,
            "first_repeat_step": first_repeat_step,
        },
    )
