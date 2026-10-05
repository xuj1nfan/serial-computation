"""自动 validator：保证任务样本与模型答案 100% 程序可验证。

分两层：
1. validate_task —— 对生成的任务样本做结构校验，并用 executor 独立复算
   ground-truth 轨迹；同时把 prompt 解析回结构化字段做 round-trip 一致性检查。
2. parse_model_answer / validate_answer —— 解析模型输出并做 exact match 评分。
"""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from src.tasks.random_dag import (
    TASK_FAMILY as DAG_TASK_FAMILY,
)
from src.tasks.random_dag import (
    RandomDagTask,
    build_dag_prompt,
    compute_graph_depths,
    execute_dag,
)
from src.tasks.state_transition import (
    TASK_FAMILY as STATE_TASK_FAMILY,
)
from src.tasks.state_transition import (
    StateTransitionTask,
    build_direct_prompt,
    execute_transition,
)

_TRANSITION_LINE = re.compile(r"^([A-Z]) -> ([A-Z])$")
_START_LINE = re.compile(r"^Starting state: ([A-Z])$")
_DEPTH_LINE = re.compile(r"^Apply the transition rule exactly (\d+) times\.$")
_FINAL_TAG = re.compile(r"Final\s*[:：]\s*([A-Za-z]+)")


Task = StateTransitionTask | RandomDagTask


def validate_state_transition_task(task: StateTransitionTask) -> list[str]:
    """校验单个状态转移样本。返回问题列表，空列表表示完全合法。"""
    problems: list[str] = []

    # --- 结构校验 ---
    if task.task_family != STATE_TASK_FAMILY:
        problems.append(f"unexpected task_family: {task.task_family!r}")
    if len(task.states) != len(set(task.states)):
        problems.append("states contain duplicates")
    if any(re.fullmatch(r"[A-Z]", state) is None for state in task.states):
        problems.append("states must be single uppercase letters")
    state_set = set(task.states)
    if set(task.transitions.keys()) != state_set:
        problems.append("transition table keys do not equal the state set")
    bad_targets = [v for v in task.transitions.values() if v not in state_set]
    if bad_targets:
        problems.append(f"transition targets outside state set: {bad_targets}")
    if task.start_state not in state_set:
        problems.append(f"start state {task.start_state!r} not in state set")
    if task.depth < 1:
        problems.append(f"depth must be >= 1, got {task.depth}")
    if len(task.trajectory) != task.depth + 1:
        problems.append(
            f"trajectory length {len(task.trajectory)} != depth+1 ({task.depth + 1})"
        )
    if task.trajectory and task.trajectory[0] != task.start_state:
        problems.append("trajectory[0] != start_state")
    unknown = [s for s in task.trajectory if s not in state_set]
    if unknown:
        problems.append(f"trajectory contains states outside state set: {unknown}")

    # --- 独立复算 ground truth（exact executor 交叉验证）---
    if not problems:
        recomputed = execute_transition(task.transitions, task.start_state, task.depth)
        if recomputed != task.trajectory:
            problems.append("trajectory does not match independent re-execution")
        if task.final_state != recomputed[-1]:
            problems.append(
                f"final_state {task.final_state!r} != recomputed {recomputed[-1]!r}"
            )

    # --- prompt round-trip：把 prompt 解析回结构化字段并比对 ---
    parsed = parse_direct_prompt(task.prompt)
    if parsed is None:
        problems.append("prompt could not be parsed back")
    else:
        p_transitions, p_start, p_depth = parsed
        if p_transitions != task.transitions:
            problems.append("prompt transition table does not match task.transitions")
        if p_start != task.start_state:
            problems.append("prompt start state does not match task.start_state")
        if p_depth != task.depth:
            problems.append("prompt depth does not match task.depth")
    expected_prompt = build_direct_prompt(
        task.states, task.transitions, task.start_state, task.depth
    )
    if task.prompt != expected_prompt:
        problems.append("prompt is not the canonical rendering of the task")

    return problems


def validate_random_dag_task(task: RandomDagTask) -> list[str]:
    """Validate graph structure, exact depth, execution, and prompt rendering."""
    problems: list[str] = []
    symbol_set = set(task.symbols)

    if task.task_family != DAG_TASK_FAMILY:
        problems.append(f"unexpected task_family: {task.task_family!r}")
    if len(task.symbols) != len(symbol_set):
        problems.append("symbols contain duplicates")
    if not symbol_set:
        problems.append("symbol set is empty")
    if any(re.fullmatch(r"[A-Z]", symbol) is None for symbol in task.symbols):
        problems.append("symbols must be single uppercase letters")
    if set(task.operator_table) != symbol_set:
        problems.append("operator table rows do not equal the symbol set")
    else:
        output_counts: Counter[str] = Counter()
        for left, row in task.operator_table.items():
            if set(row) != symbol_set:
                problems.append(f"operator table columns are incomplete for {left}")
            bad_values = [value for value in row.values() if value not in symbol_set]
            if bad_values:
                problems.append(
                    f"operator table outputs outside symbol set: {bad_values}"
                )
            output_counts.update(row.values())
        expected_count = len(task.symbols)
        if any(output_counts[symbol] != expected_count for symbol in task.symbols):
            problems.append("operator table output distribution is not balanced")

    if task.work != len(task.operations):
        problems.append(
            f"work={task.work} does not match {len(task.operations)} operations"
        )
    if len(task.inputs) != task.work + 1:
        problems.append("a full binary tree with W operations must have W+1 inputs")
    input_counts = Counter(task.inputs.values())
    if (
        input_counts
        and max(input_counts.values())
        - min(input_counts.get(symbol, 0) for symbol in task.symbols)
        > 1
    ):
        problems.append("input symbol distribution is not balanced")
    if task.output_node not in {op.target for op in task.operations}:
        problems.append("output node is not defined by an operation")

    recomputed_values: dict[str, str] | None = None
    graph_depths: dict[str, int] | None = None
    try:
        recomputed_values = execute_dag(
            task.symbols, task.operator_table, task.inputs, task.operations
        )
        graph_depths = compute_graph_depths(task.inputs, task.operations)
    except ValueError as exc:
        problems.append(str(exc))

    if graph_depths is not None and task.output_node in graph_depths:
        measured_depth = graph_depths[task.output_node]
        if measured_depth != task.depth:
            problems.append(
                f"declared depth {task.depth} != measured depth {measured_depth}"
            )

        parents = {op.target: (op.left, op.right) for op in task.operations}
        reachable: set[str] = set()
        stack = [task.output_node]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(parents.get(node, ()))
        unused_operations = set(parents) - reachable
        unused_inputs = set(task.inputs) - reachable
        if unused_operations:
            problems.append(
                f"operations do not contribute to output: {unused_operations}"
            )
        if unused_inputs:
            problems.append(f"inputs do not contribute to output: {unused_inputs}")

    if recomputed_values is not None and task.output_node in recomputed_values:
        if recomputed_values != task.node_values:
            problems.append("node_values do not match independent re-execution")
        expected_final = recomputed_values[task.output_node]
        if task.final_state != expected_final:
            problems.append(
                f"final_state {task.final_state!r} != recomputed {expected_final!r}"
            )

    if task.prompt != build_dag_prompt(task):
        problems.append("prompt is not the canonical rendering of the task")
    return problems


def validate_task(task: Task) -> list[str]:
    """Dispatch validation based on the concrete task type."""
    if isinstance(task, StateTransitionTask):
        return validate_state_transition_task(task)
    if isinstance(task, RandomDagTask):
        return validate_random_dag_task(task)
    raise TypeError(f"unsupported task type: {type(task).__name__}")


def parse_direct_prompt(
    prompt: str,
) -> tuple[dict[str, str], str, int] | None:
    """把 Direct Answer prompt 解析回 (transitions, start_state, depth)。

    解析失败返回 None。用于验证 prompt 与样本字段 100% 一致。
    """
    transitions: dict[str, str] = {}
    start_state: str | None = None
    depth: int | None = None
    for line in prompt.splitlines():
        line = line.strip()
        if m := _TRANSITION_LINE.match(line):
            transitions[m.group(1)] = m.group(2)
        elif m := _START_LINE.match(line):
            start_state = m.group(1)
        elif m := _DEPTH_LINE.match(line):
            depth = int(m.group(1))
    if not transitions or start_state is None or depth is None:
        return None
    return transitions, start_state, depth


def parse_model_answer(output: str, states: list[str]) -> dict:
    """解析模型输出中的最终状态。

    解析策略（按优先级）：
    1. "Final: X" 标签；
    2. 整个输出（去除空白标点）恰为某个状态名；
    3. 最后一个非空行的最后一个词是状态名。

    不做宽松的整词搜索：状态名是单字母，"I"/"A" 等英文词会造成假阳性。

    返回 {"answer": str|None, "parse_status": str}。
    """
    state_set = set(states)
    text = output.strip()
    if not text:
        return {"answer": None, "parse_status": "empty"}

    m = _FINAL_TAG.findall(text)
    if m and m[-1] in state_set:
        return {"answer": m[-1], "parse_status": "final_tag"}

    stripped = text.strip().strip(".").strip()
    if stripped in state_set:
        return {"answer": stripped, "parse_status": "exact"}

    nonempty_lines = [ln for ln in text.splitlines() if ln.strip()]
    tokens = re.findall(r"[A-Za-z]+", nonempty_lines[-1])
    if tokens and tokens[-1] in state_set:
        return {"answer": tokens[-1], "parse_status": "last_token"}

    return {"answer": None, "parse_status": "no_state_found"}


def validate_answer(task: Task, model_output: str) -> dict:
    """对模型输出做 exact match 评分（论文方案第 13.1 节）。"""
    parsed = parse_model_answer(model_output, task_answer_symbols(task))
    return {
        "task_id": task.task_id,
        "parsed_answer": parsed["answer"],
        "parse_status": parsed["parse_status"],
        "ground_truth": task.final_state,
        "correct": parsed["answer"] == task.final_state,
    }


def task_answer_symbols(task: Task) -> list[str]:
    if isinstance(task, StateTransitionTask):
        return task.states
    return task.symbols


def load_task(data: dict) -> Task:
    task_family = data.get("task_family")
    if task_family == STATE_TASK_FAMILY:
        return StateTransitionTask.from_dict(data)
    if task_family == DAG_TASK_FAMILY:
        return RandomDagTask.from_dict(data)
    raise ValueError(f"unknown task_family: {task_family!r}")


def validate_dataset(path: str, verbose: bool = False) -> dict:
    """校验整个 JSONL 数据集：逐条重建任务并运行 validate_task。

    返回 {"total": int, "valid": int, "invalid": [(task_id, problems), ...]}。
    """
    total = 0
    invalid: list[tuple[str, list[str]]] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            data = json.loads(line)
            task_id = str(data.get("task_id", f"line-{total}"))
            problems: list[str] = []
            if task_id in seen_ids:
                problems.append("duplicate task_id")
            seen_ids.add(task_id)
            try:
                task = load_task(data)
                problems.extend(validate_task(task))
            except (TypeError, ValueError, KeyError) as exc:
                problems.append(f"could not load task: {exc}")
            if problems:
                invalid.append((task_id, problems))
                if verbose:
                    print(f"INVALID {task_id}: {problems}")
    return {"total": total, "valid": total - len(invalid), "invalid": invalid}


def main() -> int:
    parser = ArgumentParser(description="Validate a generated JSONL dataset")
    parser.add_argument("path", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = validate_dataset(str(args.path), verbose=args.verbose)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not result["invalid"] else 1


if __name__ == "__main__":
    sys.exit(main())
