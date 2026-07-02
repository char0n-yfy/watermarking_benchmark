from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.parallel_tuning import (  # noqa: E402
    ParallelTuningService,
    TuningRequest,
    VIEWPOINT_RERENDERING_PRIMARY_METHOD,
)


class ParallelTuningPolicyTest(unittest.TestCase):
    def test_cancel_marks_running_job_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            service._write_state(
                "tune_cancel",
                {
                    "id": "tune_cancel",
                    "status": "running",
                    "progress": 12,
                    "message": "running slow candidate",
                    "events": [],
                },
            )

            cancelled = service.cancel("tune_cancel")

            self.assertEqual(cancelled["status"], "cancelled")
            self.assertTrue(cancelled["cancelRequested"])
            self.assertEqual(cancelled["message"], "tuning cancelled")
            self.assertEqual(cancelled["events"][-1]["stage"], "cancel")

    def test_fixed_attack_batch_method_is_skipped_during_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                tune_watermarks=False,
                tune_quality=False,
                attack_methods=["brightness", "2x_regen"],
            )

            methods = service._attack_methods_for_tuning(request)

            self.assertIn("brightness", methods)
            self.assertNotIn("2x_regen", methods)

    def test_fixed_attack_batch_override_is_preserved_in_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            report = {
                "jobId": "tune_test",
                "watermarks": [],
                "attacks": [],
                "quality": {},
            }

            summary = service._build_summary(report)

            self.assertIn("2x_regen=8", summary["attackBatchOverrides"])
            self.assertIn("2x_regen=8", summary["fixedAttackBatchOverrides"])
            self.assertIn("2x_regen=8", summary["envUpdates"]["WM_BENCH_ATTACK_BATCH_SIZES"])

    def test_viewpoint_rerendering_is_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                tune_watermarks=False,
                tune_quality=False,
                attack_methods=[
                    "brightness",
                    "3d_viewpoint_rerendering_swipe_ahead",
                    "3d_viewpoint_rerendering_rotate_forward_point",
                ],
            )

            methods = service._attack_methods_for_tuning(request)

            self.assertIn("brightness", methods)
            self.assertNotIn(VIEWPOINT_RERENDERING_PRIMARY_METHOD, methods)
            self.assertNotIn("3d_viewpoint_rerendering_swipe_ahead", methods)
            self.assertNotIn("3d_viewpoint_rerendering_rotate_forward_point", methods)

    def test_viewpoint_rerendering_tunes_only_primary_variant_when_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                tune_watermarks=False,
                tune_quality=False,
                include_viewpoint_3d_attacks=True,
                attack_methods=[
                    "brightness",
                    "3d_viewpoint_rerendering_swipe_ahead",
                    "3d_viewpoint_rerendering_rotate_forward_point",
                ],
            )

            methods = service._attack_methods_for_tuning(request)

            self.assertIn("brightness", methods)
            self.assertIn(VIEWPOINT_RERENDERING_PRIMARY_METHOD, methods)
            self.assertNotIn("3d_viewpoint_rerendering_swipe_ahead", methods)
            self.assertNotIn("3d_viewpoint_rerendering_rotate_forward_point", methods)
            self.assertEqual(methods.count(VIEWPOINT_RERENDERING_PRIMARY_METHOD), 1)

    def test_viewpoint_rerendering_summary_expands_primary_batch_to_all_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            report = {
                "jobId": "tune_test",
                "watermarks": [],
                "attacks": [
                    {
                        "method": VIEWPOINT_RERENDERING_PRIMARY_METHOD,
                        "bestBatch": {"batchSize": 8, "ok": True, "imagesPerSecond": 1.0},
                    }
                ],
                "quality": {},
            }

            summary = service._build_summary(report)

            overrides = summary["attackBatchOverrides"]
            self.assertIn("3d_viewpoint_rerendering_rotate_point=8", overrides)
            self.assertIn("3d_viewpoint_rerendering_rotate_ahead=8", overrides)
            self.assertIn("3d_viewpoint_rerendering_swipe_point=8", overrides)
            self.assertIn("3d_viewpoint_rerendering_shake_ahead=8", overrides)
            self.assertEqual(len([item for item in overrides if item.startswith("3d_viewpoint_rerendering_")]), 8)
            self.assertIn(
                "3d_viewpoint_rerendering_rotate_ahead=8",
                summary["inheritedAttackBatchOverrides"],
            )
            self.assertEqual(
                summary["viewpointRerenderingTuningPolicy"]["primaryMethod"],
                VIEWPOINT_RERENDERING_PRIMARY_METHOD,
            )
            self.assertIn("WM_BENCH_ATTACK_BATCH_SIZES", summary["envUpdates"])


if __name__ == "__main__":
    unittest.main()
