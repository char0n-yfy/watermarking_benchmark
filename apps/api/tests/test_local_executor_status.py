from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services import local_executor
from evaluator.attacks.base import AttackResult


class LocalExecutorStatusTest(unittest.TestCase):
    def test_stage_status_reflects_per_image_failure(self) -> None:
        result = AttackResult(
            input_path=Path("in.png"),
            output_path=Path("out.png"),
            attack_name="synthetic",
            params={},
            elapsed_ms=1.0,
            ok=False,
            error="synthetic failure",
        )

        status, error = local_executor._stage_status_and_error([result])
        self.assertEqual(status, "failed")
        self.assertEqual(error, "synthetic failure")

    def test_stage_status_uses_fallback_error_when_expected_results_are_missing(self) -> None:
        status, error = local_executor._stage_status_and_error(
            [],
            fallback_error="stage crashed",
            expected_count=1,
        )

        self.assertEqual(status, "failed")
        self.assertEqual(error, "stage crashed")

    def test_stage_status_allows_empty_expected_empty_stage(self) -> None:
        status, error = local_executor._stage_status_and_error(
            [],
            fallback_error="not used",
            expected_count=0,
        )

        self.assertEqual(status, "succeeded")
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
