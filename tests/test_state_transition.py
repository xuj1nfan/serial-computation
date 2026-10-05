from __future__ import annotations

import random
import unittest

from src.tasks.state_transition import generate_state_transition_task
from src.tasks.validators import parse_model_answer, validate_task


class StateTransitionTaskTests(unittest.TestCase):
    def test_cycle_free_generation_is_constructive(self) -> None:
        task = generate_state_transition_task(
            state_space_size=16,
            depth=15,
            rng=random.Random(0),
            task_id="cycle-free",
            avoid_cycle_within_depth=True,
        )
        self.assertEqual(len(set(task.trajectory)), 16)
        self.assertEqual(validate_task(task), [])

    def test_cycle_free_rejects_too_small_state_space(self) -> None:
        with self.assertRaises(ValueError):
            generate_state_transition_task(
                8, 8, random.Random(0), "impossible", avoid_cycle_within_depth=True
            )

    def test_answer_parser_avoids_single_letter_false_positive(self) -> None:
        parsed = parse_model_answer("I do not know", ["I", "A", "B"])
        self.assertIsNone(parsed["answer"])


if __name__ == "__main__":
    unittest.main()
