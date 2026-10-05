from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.summarize import summarize, wilson_interval
from src.generation.generate_dataset import generate_from_config, write_dataset
from src.tasks.validators import validate_dataset


class GenerationTests(unittest.TestCase):
    def test_config_generation_write_and_validation(self) -> None:
        config = {
            "task_family": "random_dag",
            "work": 7,
            "depths": [3, 5, 7],
            "symbol_count": 4,
            "n_per_cell": 2,
            "seed": 11,
            "out": "unused.jsonl",
        }
        records = generate_from_config(config)
        self.assertEqual(len(records), 6)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tasks.jsonl"
            manifest = write_dataset(records, output, config)
            self.assertEqual(manifest["records"], 6)
            self.assertEqual(validate_dataset(str(output))["valid"], 6)
            self.assertTrue(output.with_suffix(".jsonl.manifest.json").exists())

    def test_dataset_validator_detects_duplicate_ids(self) -> None:
        config = {
            "task_family": "random_dag",
            "work": 3,
            "depths": [2],
            "symbol_count": 3,
            "n_per_cell": 1,
            "seed": 1,
        }
        record = generate_from_config(config)[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "duplicate.jsonl"
            with output.open("w", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
                file.write(json.dumps(record) + "\n")
            result = validate_dataset(str(output))
        self.assertEqual(result["valid"], 1)
        self.assertIn("duplicate task_id", result["invalid"][0][1])


class AnalysisTests(unittest.TestCase):
    def test_summary_deduplicates_resumed_results(self) -> None:
        records = [
            {"status": "ok", "task_id": "a", "work": 7, "depth": 3, "correct": True},
            {"status": "ok", "task_id": "a", "work": 7, "depth": 3, "correct": False},
            {"status": "ok", "task_id": "b", "work": 7, "depth": 3, "correct": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            with path.open("w", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")
            rows = summarize(path)
        self.assertEqual(rows[0]["n"], 2)
        self.assertEqual(rows[0]["accuracy"], 0.5)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(80, 100)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)


if __name__ == "__main__":
    unittest.main()
