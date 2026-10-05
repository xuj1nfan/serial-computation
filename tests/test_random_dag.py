from __future__ import annotations

import copy
import random
import unittest
from collections import Counter

from src.tasks.random_dag import (
    compute_graph_depths,
    generate_random_dag_task,
)
from src.tasks.validators import validate_answer, validate_task


class RandomDagTaskTests(unittest.TestCase):
    def test_pilot_shapes_have_exact_work_and_depth(self) -> None:
        for depth in (5, 8, 12, 16, 24, 31):
            with self.subTest(depth=depth):
                task = generate_random_dag_task(
                    work=31,
                    depth=depth,
                    symbol_count=8,
                    rng=random.Random(depth),
                    task_id=f"depth-{depth}",
                )
                measured = compute_graph_depths(task.inputs, task.operations)
                self.assertEqual(len(task.operations), 31)
                self.assertEqual(measured[task.output_node], depth)
                self.assertEqual(validate_task(task), [])

    def test_every_operation_and_input_contributes_to_output(self) -> None:
        task = generate_random_dag_task(15, 9, 6, random.Random(4), "reachability")
        parents = {op.target: (op.left, op.right) for op in task.operations}
        reachable: set[str] = set()
        stack = [task.output_node]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(parents.get(node, ()))
        self.assertEqual(set(parents), set(parents) & reachable)
        self.assertEqual(set(task.inputs), set(task.inputs) & reachable)

    def test_operator_outputs_are_balanced(self) -> None:
        task = generate_random_dag_task(7, 3, 8, random.Random(9), "balanced")
        counts = Counter(
            value for row in task.operator_table.values() for value in row.values()
        )
        self.assertEqual(set(counts.values()), {8})
        input_counts = Counter(task.inputs.values())
        self.assertLessEqual(max(input_counts.values()) - min(input_counts.values()), 1)

    def test_generation_is_reproducible(self) -> None:
        first = generate_random_dag_task(15, 7, 5, random.Random(12), "same")
        second = generate_random_dag_task(15, 7, 5, random.Random(12), "same")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_validator_detects_tampering(self) -> None:
        task = generate_random_dag_task(7, 4, 4, random.Random(2), "tampered")
        task = copy.deepcopy(task)
        task.depth += 1
        self.assertTrue(
            any("measured depth" in problem for problem in validate_task(task))
        )

    def test_answer_scoring_works_for_dag(self) -> None:
        task = generate_random_dag_task(7, 4, 4, random.Random(3), "answer")
        result = validate_answer(task, task.final_state)
        self.assertTrue(result["correct"])

    def test_rejects_infeasible_shape(self) -> None:
        with self.assertRaises(ValueError):
            generate_random_dag_task(31, 4, 8, random.Random(0), "bad")


if __name__ == "__main__":
    unittest.main()
