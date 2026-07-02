from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.parallel_tuning import (  # noqa: E402
    DIFFUSION_REGENERATION_PRIMARY_METHOD,
    ParallelTuningService,
    TuningRequest,
    VIEWPOINT_RERENDERING_PRIMARY_METHOD,
)


class ParallelTuningPolicyTest(unittest.TestCase):
    def test_tuning_request_preserves_selected_methods(self) -> None:
        request = TuningRequest.from_payload(
            {
                "tuneWatermarks": True,
                "tuneAttacks": True,
                "watermarkMethods": ["dwsf", "trustmark-q"],
                "attackMethods": ["jpeg", "brightness"],
            }
        )

        self.assertEqual(request.watermark_methods, ["dwsf", "trustmark-q"])
        self.assertEqual(request.attack_methods, ["jpeg", "brightness"])
        self.assertEqual(request.to_json()["watermarkMethods"], ["dwsf", "trustmark-q"])
        self.assertEqual(request.to_json()["attackMethods"], ["jpeg", "brightness"])

    def test_cancel_requests_running_job_stop_without_marking_terminal(self) -> None:
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
            release = threading.Event()
            thread = threading.Thread(target=release.wait, daemon=True)
            thread.start()
            service._threads["tune_cancel"] = thread

            try:
                cancelled = service.cancel("tune_cancel")
            finally:
                release.set()
                thread.join(timeout=1)

            self.assertEqual(cancelled["status"], "cancelling")
            self.assertTrue(cancelled["cancelRequested"])
            self.assertNotIn("finishedAt", cancelled)
            self.assertEqual(cancelled["message"], "tuning cancellation requested")
            self.assertEqual(cancelled["events"][-1]["stage"], "cancel_requested")

    def test_start_is_blocked_while_job_is_cancelling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            service._write_state(
                "tune_cancelling",
                {
                    "id": "tune_cancelling",
                    "status": "cancelling",
                    "progress": 55,
                    "message": "waiting for current candidate to finish",
                    "events": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "already running"):
                service.start({"tuneWatermarks": False, "tuneAttacks": False, "tuneQuality": False})

    def test_start_is_blocked_while_tracked_thread_is_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            service._write_state(
                "tune_old_cancel",
                {
                    "id": "tune_old_cancel",
                    "status": "cancelled",
                    "progress": 55,
                    "cancelRequested": True,
                    "message": "old terminal state but worker still exiting",
                    "events": [],
                },
            )
            release = threading.Event()
            thread = threading.Thread(target=release.wait, daemon=True)
            thread.start()
            service._threads["tune_old_cancel"] = thread

            try:
                with self.assertRaisesRegex(ValueError, "still stopping"):
                    service.start({"tuneWatermarks": False, "tuneAttacks": False, "tuneQuality": False})
            finally:
                release.set()
                thread.join(timeout=1)

    def test_start_reports_running_when_live_thread_state_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            service._write_state(
                "tune_running",
                {
                    "id": "tune_running",
                    "status": "running",
                    "progress": 20,
                    "message": "running candidate",
                    "events": [],
                },
            )
            release = threading.Event()
            thread = threading.Thread(target=release.wait, daemon=True)
            thread.start()
            service._threads["tune_running"] = thread

            try:
                with self.assertRaisesRegex(ValueError, "already running"):
                    service.start({"tuneWatermarks": False, "tuneAttacks": False, "tuneQuality": False})
            finally:
                release.set()
                thread.join(timeout=1)

    def test_diffusion_regeneration_tunes_only_primary_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                tune_watermarks=False,
                tune_quality=False,
                attack_methods=["brightness", "2x_regen", "4x_regen", DIFFUSION_REGENERATION_PRIMARY_METHOD],
            )

            methods = service._attack_methods_for_tuning(request)

            self.assertIn("brightness", methods)
            self.assertIn(DIFFUSION_REGENERATION_PRIMARY_METHOD, methods)
            self.assertNotIn("2x_regen", methods)
            self.assertNotIn("4x_regen", methods)
            self.assertEqual(methods.count(DIFFUSION_REGENERATION_PRIMARY_METHOD), 1)

    def test_diffusion_regeneration_summary_expands_primary_batch_to_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            report = {
                "jobId": "tune_test",
                "watermarks": [],
                "attacks": [
                    {
                        "method": DIFFUSION_REGENERATION_PRIMARY_METHOD,
                        "bestBatch": {"batchSize": 8, "ok": True, "imagesPerSecond": 1.0},
                    }
                ],
                "quality": {},
            }

            summary = service._build_summary(report)

            overrides = summary["attackBatchOverrides"]
            self.assertIn("regen_diffusion=8", overrides)
            self.assertIn("2x_regen=8", overrides)
            self.assertIn("4x_regen=8", overrides)
            self.assertIn("2x_regen=8", summary["inheritedAttackBatchOverrides"])
            self.assertIn("4x_regen=8", summary["inheritedAttackBatchOverrides"])
            self.assertEqual(
                summary["diffusionRegenerationTuningPolicy"]["primaryMethod"],
                DIFFUSION_REGENERATION_PRIMARY_METHOD,
            )

    def test_summary_omits_diffusion_regeneration_overrides_without_measurement(self) -> None:
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

            self.assertNotIn("WM_BENCH_ATTACK_BATCH_SIZES", summary["envUpdates"])
            self.assertEqual(summary["attackBatchOverrides"], [])
            self.assertEqual(summary["fixedAttackBatchOverrides"], [])

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
