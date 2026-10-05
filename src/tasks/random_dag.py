"""Random computation DAG tasks with independently controlled work and depth.

The generated graph is a full binary computation tree (and therefore a DAG).
Every internal node is one lookup-table operation, every operation contributes to
the output, and the longest dependency path is exactly ``depth``.
"""

from __future__ import annotations

import random
import string
from dataclasses import asdict, dataclass, field

TASK_FAMILY = "random_dag"
SYMBOL_POOL = list(string.ascii_uppercase)


@dataclass(frozen=True)
class DagOperation:
    target: str
    left: str
    right: str


@dataclass
class RandomDagTask:
    task_id: str
    task_family: str
    symbols: list[str]
    operator_table: dict[str, dict[str, str]]
    inputs: dict[str, str]
    operations: list[DagOperation]
    output_node: str
    work: int
    depth: int
    node_values: dict[str, str]
    final_state: str
    prompt: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RandomDagTask:
        values = dict(data)
        values["operations"] = [DagOperation(**op) for op in data["operations"]]
        return cls(**values)


@dataclass
class _TreeNode:
    left: _TreeNode | None = None
    right: _TreeNode | None = None
    name: str | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _validate_shape_parameters(work: int, depth: int) -> None:
    if work < 1:
        raise ValueError("work must be >= 1")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth > work:
        raise ValueError(f"depth ({depth}) cannot exceed work ({work})")
    if work > 2**depth - 1:
        raise ValueError(
            f"work={work} cannot fit in a binary computation tree of depth={depth}"
        )


def _leaf_slots(
    node: _TreeNode,
    current_depth: int,
    max_depth: int,
) -> list[tuple[_TreeNode, str]]:
    slots: list[tuple[_TreeNode, str]] = []
    if node.is_leaf:
        return slots
    for side in ("left", "right"):
        child = getattr(node, side)
        assert child is not None
        if child.is_leaf:
            if current_depth + 1 < max_depth:
                slots.append((node, side))
        else:
            slots.extend(_leaf_slots(child, current_depth + 1, max_depth))
    return slots


def _make_tree(work: int, depth: int, rng: random.Random) -> _TreeNode:
    """Build a randomized full binary tree with exact internal count and height."""
    _validate_shape_parameters(work, depth)

    # First create one guaranteed critical path of exactly ``depth`` operations.
    root = _TreeNode()
    for _ in range(depth):
        sibling = _TreeNode()
        if rng.randrange(2):
            root = _TreeNode(left=root, right=sibling)
        else:
            root = _TreeNode(left=sibling, right=root)

    # Expanding a leaf adds exactly one operation and never changes the maximum
    # depth. Feasibility above guarantees that a slot exists until ``work``.
    for _ in range(work - depth):
        slots = _leaf_slots(root, 0, depth)
        if not slots:
            raise RuntimeError("no expandable leaf despite feasible shape parameters")
        parent, side = rng.choice(slots)
        setattr(parent, side, _TreeNode(left=_TreeNode(), right=_TreeNode()))
    return root


def execute_dag(
    symbols: list[str],
    operator_table: dict[str, dict[str, str]],
    inputs: dict[str, str],
    operations: list[DagOperation],
) -> dict[str, str]:
    """Evaluate a topologically ordered operation list and return all node values."""
    symbol_set = set(symbols)
    values = dict(inputs)
    if any(value not in symbol_set for value in values.values()):
        raise ValueError("input value outside symbol set")

    for op in operations:
        if op.target in values:
            raise ValueError(f"duplicate node definition: {op.target}")
        if op.left not in values or op.right not in values:
            raise ValueError(f"operation {op.target} references an undefined node")
        left_value = values[op.left]
        right_value = values[op.right]
        try:
            values[op.target] = operator_table[left_value][right_value]
        except KeyError as exc:
            raise ValueError(
                f"operator table has no entry for ({left_value}, {right_value})"
            ) from exc
    return values


def compute_graph_depths(
    inputs: dict[str, str], operations: list[DagOperation]
) -> dict[str, int]:
    """Independently compute each node's dependency depth."""
    depths = {name: 0 for name in inputs}
    for op in operations:
        if op.target in depths:
            raise ValueError(f"duplicate node definition: {op.target}")
        if op.left not in depths or op.right not in depths:
            raise ValueError(f"operation {op.target} references an undefined node")
        depths[op.target] = max(depths[op.left], depths[op.right]) + 1
    return depths


def build_dag_prompt(task: RandomDagTask) -> str:
    lines = [
        "You are given a lookup-defined binary operator g and a computation graph.",
        "Use the table exactly; do not assume that g has any algebraic properties.",
        "",
        "Operator table:",
    ]
    for left in task.symbols:
        for right in task.symbols:
            lines.append(f"g({left}, {right}) = {task.operator_table[left][right]}")

    lines.extend(["", "Inputs:"])
    for name, value in task.inputs.items():
        lines.append(f"{name} = {value}")

    lines.extend(["", "Definitions:"])
    for op in task.operations:
        lines.append(f"{op.target} = g({op.left}, {op.right})")

    lines.extend(
        [
            "",
            f"Output node: {task.output_node}",
            "Return only the value of the output node.",
        ]
    )
    return "\n".join(lines)


def generate_random_dag_task(
    work: int,
    depth: int,
    symbol_count: int,
    rng: random.Random,
    task_id: str,
) -> RandomDagTask:
    """Generate one task with exactly ``work`` operations and critical depth."""
    _validate_shape_parameters(work, depth)
    if not 2 <= symbol_count <= len(SYMBOL_POOL):
        raise ValueError(f"symbol_count must be in [2, {len(SYMBOL_POOL)}]")

    symbols = rng.sample(SYMBOL_POOL, symbol_count)
    # Every symbol appears equally often in the lookup outputs. This holds the
    # primitive operator distribution constant across depth conditions.
    table_outputs = symbols * symbol_count
    rng.shuffle(table_outputs)
    output_iterator = iter(table_outputs)
    operator_table = {
        left: {right: next(output_iterator) for right in symbols} for left in symbols
    }
    tree = _make_tree(work, depth, rng)

    leaves: list[_TreeNode] = []

    def collect_leaves(node: _TreeNode) -> None:
        if node.is_leaf:
            leaves.append(node)
            return
        assert node.left is not None and node.right is not None
        collect_leaves(node.left)
        collect_leaves(node.right)

    collect_leaves(tree)
    rng.shuffle(leaves)
    repeats, remainder = divmod(len(leaves), symbol_count)
    input_values = symbols * repeats + rng.sample(symbols, remainder)
    rng.shuffle(input_values)
    inputs: dict[str, str] = {}
    for index, (leaf, value) in enumerate(zip(leaves, input_values, strict=True)):
        leaf.name = f"x{index:02d}"
        inputs[leaf.name] = value

    operations: list[DagOperation] = []

    def emit_postorder(node: _TreeNode) -> str:
        if node.is_leaf:
            assert node.name is not None
            return node.name
        assert node.left is not None and node.right is not None
        left_name = emit_postorder(node.left)
        right_name = emit_postorder(node.right)
        node.name = f"t{len(operations):02d}"
        operations.append(DagOperation(node.name, left_name, right_name))
        return node.name

    output_node = emit_postorder(tree)
    node_values = execute_dag(symbols, operator_table, inputs, operations)
    graph_depths = compute_graph_depths(inputs, operations)

    task = RandomDagTask(
        task_id=task_id,
        task_family=TASK_FAMILY,
        symbols=symbols,
        operator_table=operator_table,
        inputs=inputs,
        operations=operations,
        output_node=output_node,
        work=work,
        depth=depth,
        node_values=node_values,
        final_state=node_values[output_node],
        prompt="",
        metadata={
            "symbol_count": symbol_count,
            "graph_type": "full_binary_tree",
            "prompt_format": "direct_answer",
            "measured_depth": graph_depths[output_node],
        },
    )
    task.prompt = build_dag_prompt(task)
    return task
