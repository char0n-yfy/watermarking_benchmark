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
                "tuneQuality": True,
                "watermarkMethods": ["dwsf", "trustmark-q"],
                "attackMethods": ["jpeg", "brightness"],
                "qualityMetrics": ["psnr", "lpips"],
            }
        )

        self.assertEqual(request.watermark_methods, ["dwsf", "trustmark-q"])
        self.assertEqual(request.attack_methods, ["jpeg", "brightness"])
        self.assertEqual(request.quality_metrics, ["psnr", "lpips"])
        self.assertEqual(request.to_json()["watermarkMethods"], ["dwsf", "trustmark-q"])
        self.assertEqual(request.to_json()["attackMethods"], ["jpeg", "brightness"])
        self.assertEqual(request.to_json()["qualityMetrics"], ["psnr", "lpips"])

    def test_tuning_request_expands_samples_for_candidate_batches(self) -> None:
        request = TuningRequest.from_payload(
            {
                "sampleCount": 64,
                "candidateBatchCount": 3,
                "batchCandidates": [1, 2, 4, 8, 16, 32, 64],
                "maxBatchSize": 64,
            }
        )

        self.assertEqual(request.sample_count, 192)
        self.assertEqual(request.candidate_batch_count, 3)
        self.assertEqual(request.to_json()["candidateBatchCount"], 3)

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

    def test_identity_attack_is_not_tuned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                tune_watermarks=False,
                tune_quality=False,
                attack_methods=["identity", "brightness"],
            )

            methods = service._attack_methods_for_tuning(request)

            self.assertEqual(methods, ["brightness"])

    def test_adaptive_search_probes_all_candidates_then_remeasures_finalists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                search_strategy="adaptive",
                sample_count=10,
                probe_sample_count=4,
                finalist_count=2,
                repeat_count=3,
                batch_candidates=[1, 2, 4],
            )
            calls: list[tuple[int, int, str]] = []
            entries: list[dict[str, object]] = []

            def run_once(candidate: int, sample_count: int, phase: str) -> dict[str, object]:
                calls.append((candidate, sample_count, phase))
                return {
                    "batchSize": candidate,
                    "measurementPhase": phase,
                    "sampleCount": sample_count,
                    "elapsedSeconds": sample_count / float(candidate),
                    "imagesPerSecond": float(candidate),
                    "ok": True,
                }

            service._search_numeric_candidates(
                job_id="tune_adaptive",
                request=request,
                initial_candidates=[1, 2, 4],
                max_value=4,
                value_key="batchSize",
                run_once=run_once,
                on_entry=entries.append,
            )

            self.assertEqual(calls[:3], [(1, 3, "probe"), (2, 6, "probe"), (4, 10, "probe")])
            self.assertEqual(calls[3:], [(2, 6, "final")] * 3 + [(4, 10, "final")] * 3)
            self.assertEqual([entry["measurementPhase"] for entry in entries], ["probe", "probe", "probe", "final", "final"])

    def test_single_pass_batch_search_measures_fixed_candidate_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                sample_count=10,
                candidate_batch_count=3,
                probe_sample_count=4,
                finalist_count=2,
                repeat_count=3,
                batch_candidates=[1, 2, 4],
            )
            calls: list[tuple[int, int, str]] = []
            entries: list[dict[str, object]] = []

            def run_once(candidate: int, sample_count: int, phase: str) -> dict[str, object]:
                calls.append((candidate, sample_count, phase))
                return {
                    "batchSize": candidate,
                    "measurementPhase": phase,
                    "sampleCount": sample_count,
                    "elapsedSeconds": sample_count / float(candidate),
                    "imagesPerSecond": float(candidate),
                    "ok": True,
                }

            service._search_numeric_candidates(
                job_id="tune_single",
                request=request,
                initial_candidates=[1, 2, 4],
                max_value=4,
                value_key="batchSize",
                run_once=run_once,
                on_entry=entries.append,
            )

            self.assertEqual(calls, [(1, 3, "single_pass"), (2, 6, "single_pass"), (4, 10, "single_pass")])
            self.assertEqual([entry["measurementPhase"] for entry in entries], ["single_pass", "single_pass", "single_pass"])

    def test_single_pass_worker_search_still_uses_full_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(sample_count=10, worker_candidates=[1, 2, 4])
            calls: list[tuple[int, int, str]] = []

            def run_once(candidate: int, sample_count: int, phase: str) -> dict[str, object]:
                calls.append((candidate, sample_count, phase))
                return {
                    "workers": candidate,
                    "measurementPhase": phase,
                    "sampleCount": sample_count,
                    "elapsedSeconds": sample_count / float(candidate),
                    "imagesPerSecond": float(candidate),
                    "ok": True,
                }

            service._search_numeric_candidates(
                job_id="tune_workers",
                request=request,
                initial_candidates=[1, 2, 4],
                max_value=4,
                value_key="workers",
                run_once=run_once,
                on_entry=lambda entry: None,
            )

            self.assertEqual(calls, [(1, 10, "single_pass"), (2, 10, "single_pass"), (4, 10, "single_pass")])

    def test_incremental_persist_merges_named_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env.autodl"
            env_path.write_text(
                "WM_BENCH_ATTACK_BATCH_SIZES=cew_s2=4,noise_to_image=1\n",
                encoding="utf-8",
            )
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs", env_path=env_path)

            service._persist_env_updates(
                "tune_partial",
                {"WM_BENCH_ATTACK_BATCH_SIZES": "noise_to_image=2"},
                partial=True,
            )

            contents = env_path.read_text(encoding="utf-8")
            self.assertIn("WM_BENCH_ATTACK_BATCH_SIZES=cew_s2=4,noise_to_image=2", contents)

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

    def test_quality_summary_uses_metric_specific_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            report = {
                "jobId": "tune_test",
                "watermarks": [],
                "attacks": [],
                "quality": {
                    "bestCpuWorkers": {"workers": 16, "ok": True, "imagesPerSecond": 10.0},
                    "bestCpuWorkersByMetric": {
                        "psnr": {"workers": 8, "ok": True, "imagesPerSecond": 11.0},
                        "ssim": {"workers": 12, "ok": True, "imagesPerSecond": 12.0},
                        "ms_ssim": {"workers": 16, "ok": True, "imagesPerSecond": 13.0},
                        "nmi": {"workers": 4, "ok": True, "imagesPerSecond": 14.0},
                    },
                    "bestPerceptualBatch": {"batchSize": 16, "ok": True, "imagesPerSecond": 20.0},
                    "bestPerceptualBatchByMetric": {
                        "lpips": {"batchSize": 32, "ok": True, "imagesPerSecond": 30.0},
                        "dists": {"batchSize": 8, "ok": True, "imagesPerSecond": 15.0},
                    },
                },
            }

            summary = service._build_summary(report)

            self.assertEqual(
                summary["envUpdates"]["WM_BENCH_QUALITY_CPU_WORKERS_BY_METRIC"],
                "psnr=8,ssim=12,ms_ssim=16,nmi=4",
            )
            self.assertNotIn("WM_BENCH_QUALITY_CPU_WORKERS", summary["envUpdates"])
            self.assertEqual(summary["envUpdates"]["WM_BENCH_PERCEPTUAL_BATCH_SIZES"], "lpips=32,dists=8")
            self.assertNotIn("WM_BENCH_PERCEPTUAL_BATCH_SIZE", summary["envUpdates"])
            self.assertEqual(summary["qualityCpuWorkerOverrides"], ["psnr=8", "ssim=12", "ms_ssim=16", "nmi=4"])
            self.assertEqual(summary["qualityPerceptualBatchOverrides"], ["lpips=32", "dists=8"])

    def test_quality_step_estimate_counts_selected_metric_searches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = ParallelTuningService(resources_root=root / "resources", runs_root=root / "runs")
            request = TuningRequest(
                tune_watermarks=False,
                tune_attacks=False,
                tune_quality=True,
                batch_candidates=[1, 2, 4],
                worker_candidates=[1, 2],
                max_batch_size=4,
                max_worker_count=2,
                quality_metrics=["psnr", "lpips", "dists"],
            )

            self.assertEqual(service._estimate_steps(request), 1 + 2 + (2 * 3))


if __name__ == "__main__":
    unittest.main()
