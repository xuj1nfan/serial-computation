from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.generation.generate_dataset import write_dataset
from src.inference.run_experiment import ChatClient, run
from src.tasks.random_dag import generate_random_dag_task


class InferenceRunnerTests(unittest.TestCase):
    def test_request_scoring_and_resume(self) -> None:
        task = generate_random_dag_task(3, 2, 3, random.Random(5), "api-task")
        response = {
            "choices": [
                {
                    "message": {"content": task.final_state},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
        with (
            patch.object(ChatClient, "complete", return_value=(response, 0.01)) as mock,
            tempfile.TemporaryDirectory() as directory,
        ):
            dataset = Path(directory) / "tasks.jsonl"
            output = Path(directory) / "results.jsonl"
            write_dataset([task.to_dict()], dataset, {"test": True})
            config = {
                "dataset": str(dataset),
                "output": str(output),
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "mock-model",
                "concurrency": 1,
                "retries": 0,
            }
            self.assertEqual(run(config), 0)
            self.assertEqual(run(config), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            changed_config = {**config, "model": "different-model"}
            with self.assertRaisesRegex(ValueError, "experiment settings differ"):
                run(changed_config)

        self.assertEqual(mock.call_count, 1)
        self.assertTrue(result["correct"])
        self.assertEqual(result["parsed_answer"], task.final_state)


if __name__ == "__main__":
    unittest.main()
