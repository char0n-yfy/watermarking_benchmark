from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.local_runner import LocalRunRequest, run_local_experiment


class LocalRunnerTest(unittest.TestCase):
    def test_global_pipeline_stage_order_across_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources_root = root / "resources"
            runs_root = root / "runs"
            for dataset_id, color in (("smoke_a", (120, 160, 200)), ("smoke_b", (90, 140, 210))):
                dataset_dir = resources_root / "datasets" / dataset_id
                dataset_dir.mkdir(parents=True)
                Image.new("RGB", (300, 300), color).save(dataset_dir / "sample.png")

            summary = run_local_experiment(
                LocalRunRequest(
                    run_id="run_global_order",
                    selection={
                        "datasetIds": ["smoke_a", "smoke_b"],
                        "algorithmIds": ["alg-traditional-spread-dct"],
                        "attackPresetIds": ["atk-identity"],
                        "seeds": [42],
                        "maxSamples": 1,
                    },
                    resources_root=resources_root,
                    runs_root=runs_root,
                )
            )

            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["resultUnitCount"], 2)
            run_state = json.loads((runs_root / "run_global_order" / "run_state.json").read_text())
            phase_state = json.loads((runs_root / "run_global_order" / "phase_state.json").read_text())
            artifact_tree = json.loads((runs_root / "run_global_order" / "artifact_tree.json").read_text())

            self.assertEqual(run_state["currentPhase"], "summary")
            self.assertEqual(run_state["progressKind"], "phaseOperations")
            self.assertEqual([phase["key"] for phase in phase_state["phases"]], [
                "canonical",
                "watermark_embed",
                "attack",
                "watermark_extract",
                "quality",
                "summary",
            ])
            self.assertIn("smoke_a", artifact_tree["datasets"])
            self.assertIn("canonical", artifact_tree["datasets"]["smoke_a"])
            self.assertFalse((runs_root / "run_global_order" / "stage_events.jsonl").exists())
            self.assertFalse((runs_root / "run_global_order" / "cell_manifest.jsonl").exists())
            self.assertTrue(all(phase["status"] == "succeeded" for phase in phase_state["phases"]))

    def test_smoke_run_extracts_default_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 200), (120, 160, 200)).save(dataset_dir / "sample.png")

            with patch.dict(os.environ, {"WM_BENCH_CACHE_ROOT": ""}):
                summary = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_smoke",
                        selection={
                            "datasetIds": ["smoke"],
                            "algorithmIds": ["alg-traditional-spread-dct"],
                            "attackPresetIds": ["atk-identity"],
                            "seeds": [42],
                            "maxSamples": 1,
                        },
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
            )

            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["resultUnitCount"], 1)
            result_unit = summary["resultUnits"][0]
            self.assertEqual(result_unit["status"], "succeeded")
            extract_manifest = json.loads(Path(result_unit["manifestPath"]).read_text())
            self.assertEqual(extract_manifest[0]["decodedBits"], extract_manifest[0]["expectedBits"])
            self.assertEqual(extract_manifest[0]["metadata"]["decodeInputSize"], [512, 512])
            self.assertEqual(extract_manifest[0]["metadata"]["decodeInternalSize"], [512, 512])
            self.assertNotIn("bitAccuracy", result_unit)
            self.assertNotIn("bitErrorRate", result_unit)
            self.assertNotIn("scoring", result_unit)
            self.assertNotIn("score", summary)
            self.assertNotIn("aggregates", summary)
            self.assertNotIn("bit_accuracy", extract_manifest[0]["metadata"])
            self.assertNotIn("match", extract_manifest[0]["metadata"])
            self.assertNotIn("inputPath", extract_manifest[0])
            self.assertNotIn("payloadBits", extract_manifest[0])
            self.assertNotIn("detectionScore", extract_manifest[0])
            self.assertNotIn("negativeManifestPath", result_unit)
            run_root = runs_root / "run_smoke"
            materialized = result_unit["materialized"]
            self.assertTrue((run_root / "run_summary.json").exists())
            self.assertTrue((run_root / "run_plan.json").exists())
            self.assertTrue((run_root / "run_state.json").exists())
            self.assertTrue((run_root / "phase_state.json").exists())
            self.assertTrue((run_root / "artifact_tree.json").exists())
            self.assertTrue((run_root / "result_units.jsonl").exists())
            self.assertTrue((run_root / "image_quality.jsonl").exists())
            self.assertTrue((run_root / "image_watermark_embed.jsonl").exists())
            self.assertTrue((run_root / "image_attack.jsonl").exists())
            self.assertTrue((run_root / "image_detection.jsonl").exists())
            self.assertTrue((run_root / "runtime_profile.jsonl").exists())
            self.assertFalse((run_root / "cell_manifest.jsonl").exists())
            self.assertFalse((run_root / "cell_summary_latest.json").exists())
            self.assertFalse((run_root / "stage_events.jsonl").exists())
            self.assertEqual(summary["materializedRoot"], str(run_root / "materialized"))
            self.assertTrue(Path(materialized["canonicalDir"]).exists())
            self.assertTrue(Path(materialized["watermarkedDir"]).exists())
            self.assertTrue(Path(materialized["attackedDir"]).exists())
            self.assertTrue(Path(materialized["negativeAttackedDir"]).exists())
            run_status = json.loads((run_root / "run_status.json").read_text())
            run_plan = json.loads((run_root / "run_plan.json").read_text())
            self.assertEqual(run_status["progress"], 100)
            self.assertEqual(run_status["completedProgress"], 100)
            self.assertEqual(run_status["progressKind"], "phaseOperations")
            self.assertEqual(run_status["resultUnitCount"], 1)
            self.assertEqual(summary["completedProgress"], 100)
            self.assertEqual(summary["succeededProgress"], 100)
            self.assertEqual(summary["progressKind"], "phaseOperations")
            sample_record = json.loads((run_root / "sample_manifest.jsonl").read_text().splitlines()[0])
            quality_record = json.loads((run_root / "image_quality.jsonl").read_text().splitlines()[0])
            runtime_record = json.loads((run_root / "runtime_profile.jsonl").read_text().splitlines()[0])
            result_unit_record = json.loads((run_root / "result_units.jsonl").read_text().splitlines()[0])
            self.assertIn("executionPolicy", run_plan)
            self.assertIn("execution", runtime_record["metadata"])
            self.assertEqual(result_unit_record["resultUnitKey"], result_unit["resultUnitKey"])
            self.assertNotIn("stagedPath", sample_record)
            self.assertEqual(sample_record["originalSize"], [300, 200])
            self.assertEqual(sample_record["canonicalSize"], [512, 512])
            self.assertEqual(sample_record["preprocessPolicy"], "center_cover_crop_512")
            self.assertEqual(sample_record["cropPolicy"], "deterministic_center_cover_crop")
            self.assertEqual(sample_record["resizedContentSize"], [768, 512])
            self.assertEqual(sample_record["cropBox"], [128, 0, 640, 512])
            self.assertEqual(sample_record["cropMargins"], {"left": 128, "top": 0, "right": 128, "bottom": 0})
            self.assertIsNone(sample_record["padding"])
            self.assertIsNone(sample_record["paddingColor"])
            self.assertEqual(quality_record["referenceSize"], [512, 512])
            self.assertEqual(quality_record["targetSize"], [512, 512])
            self.assertEqual(quality_record["alignedSize"], [512, 512])
            self.assertEqual(quality_record["alignmentPolicy"], "none")
            for field in ("width", "height", "referencePath", "targetPath"):
                self.assertNotIn(field, quality_record)
            for field in ("perceptualBackend", "perceptualDevice", "perceptualErrors"):
                self.assertNotIn(field, quality_record["metrics"])
            for field in ("msPerImage", "msPerMP", "throughputImagesPerSecond", "throughputMPPerSecond", "macs", "flops"):
                self.assertNotIn(field, runtime_record)
            self.assertFalse((Path(result_unit["outputDir"]) / "attacked").exists())

            runtime_elapsed_before_resume = sum(
                float(json.loads(line).get("elapsedMs") or 0.0)
                for line in (run_root / "runtime_profile.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            summary_path = run_root / "run_summary.json"
            stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            stored_summary["elapsedMs"] = 1.0
            summary_path.write_text(json.dumps(stored_summary), encoding="utf-8")

            with patch.dict(os.environ, {"WM_BENCH_CACHE_ROOT": ""}):
                resumed = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_smoke",
                        selection={
                            "datasetIds": ["smoke"],
                            "algorithmIds": ["alg-traditional-spread-dct"],
                            "attackPresetIds": ["atk-identity"],
                            "seeds": [42],
                            "maxSamples": 1,
                        },
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
            )
            self.assertEqual(resumed["status"], "succeeded")
            self.assertEqual(resumed["resultUnitCount"], 1)
            self.assertEqual(resumed["skippedResultUnits"], 1)
            self.assertGreaterEqual(resumed["elapsedMs"], runtime_elapsed_before_resume)

            resumed_phase_state = json.loads((run_root / "phase_state.json").read_text(encoding="utf-8"))
            phase_by_key = {phase["key"]: phase for phase in resumed_phase_state["phases"]}
            for phase_key in (
                "canonical",
                "watermark_embed",
                "attack",
                "watermark_extract",
                "quality",
                "summary",
            ):
                self.assertEqual(phase_by_key[phase_key]["current"], phase_by_key[phase_key]["total"])
            self.assertEqual(phase_by_key["canonical"]["counters"]["imagesDone"], 1)
            self.assertEqual(phase_by_key["watermark_embed"]["counters"]["imagesDone"], 1)
            self.assertEqual(phase_by_key["attack"]["counters"]["positiveImagesDone"], 1)
            self.assertEqual(phase_by_key["attack"]["counters"]["negativeImagesDone"], 1)
            self.assertEqual(phase_by_key["watermark_extract"]["counters"]["imagesDone"], 2)
            self.assertEqual(phase_by_key["quality"]["counters"]["pairsDone"], 2)
            self.assertEqual(phase_by_key["summary"]["counters"]["skippedUnits"], 1)

    def test_negative_attack_outputs_are_reused_across_algorithms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (90, 140, 210)).save(dataset_dir / "sample.png")

            from app.services import local_executor

            original_catalog_item = local_executor.get_watermark_catalog_item

            def catalog_item(identifier: str):
                if identifier == "alg-traditional-spread-dct-copy":
                    item = dict(original_catalog_item("alg-traditional-spread-dct"))
                    item["id"] = identifier
                    return item
                return original_catalog_item(identifier)

            with patch("app.services.local_executor.get_watermark_catalog_item", side_effect=catalog_item):
                summary = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_negative_reuse",
                        selection={
                            "datasetIds": ["smoke"],
                            "algorithmIds": [
                                "alg-traditional-spread-dct",
                                "alg-traditional-spread-dct-copy",
                            ],
                            "attackPresetIds": ["atk-identity"],
                            "seeds": [42],
                            "maxSamples": 1,
                        },
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
            )

            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["resultUnitCount"], 2)
            self.assertEqual(
                summary["resultUnits"][0]["materialized"]["negativeAttackedDir"],
                summary["resultUnits"][1]["materialized"]["negativeAttackedDir"],
            )

            run_root = runs_root / "run_negative_reuse"
            profiles = [
                json.loads(line)
                for line in (run_root / "runtime_profile.jsonl").read_text().splitlines()
                if line.strip()
            ]
            negative_profiles = [
                record
                for record in profiles
                if record["stage"] == "attack_negative_control"
            ]
            self.assertEqual(len(negative_profiles), 2)
            self.assertEqual(
                sum(1 for record in negative_profiles if record.get("metadata", {}).get("cacheHit") is True),
                1,
            )
            self.assertEqual(
                sum(1 for record in negative_profiles if record.get("status") == "reused"),
                1,
            )
            self.assertFalse((run_root / "staging" / "negative_attacked" / "smoke").exists())

    def test_global_materialized_cache_is_reused_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            cache_root = root / "cache" / "materialized"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 200), (120, 160, 200)).save(dataset_dir / "sample.png")

            selection = {
                "datasetIds": ["smoke"],
                "algorithmIds": ["alg-traditional-spread-dct"],
                "attackPresetIds": ["atk-identity"],
                "seeds": [42],
                "maxSamples": 1,
            }
            with patch.dict(os.environ, {"WM_BENCH_CACHE_ROOT": str(cache_root)}):
                first = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_cache_first",
                        selection=selection,
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
                )
                second = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_cache_second",
                        selection=selection,
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
                )

            self.assertEqual(first["status"], "succeeded")
            self.assertEqual(second["status"], "succeeded")
            self.assertEqual(second["materializedRoot"], str(cache_root))
            profiles = [
                json.loads(line)
                for line in (runs_root / "run_cache_second" / "runtime_profile.jsonl").read_text().splitlines()
                if line.strip()
            ]
            reused_stages = {
                record["stage"]
                for record in profiles
                if record["status"] == "reused"
                and record.get("metadata", {}).get("cacheHit") is True
            }
            self.assertIn("watermark_embed", reused_stages)
            self.assertIn("attack", reused_stages)
            self.assertIn("attack_negative_control", reused_stages)

    def test_materialized_cache_reuses_sample_prefix_across_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            cache_root = root / "cache" / "materialized"
            dataset_dir.mkdir(parents=True)
            for index in range(4):
                Image.new("RGB", (96, 96), (80 + index, 120, 180)).save(dataset_dir / f"{index:03d}.png")

            base_selection = {
                "datasetIds": ["smoke"],
                "algorithmIds": ["alg-traditional-spread-dct"],
                "attackPresetIds": ["atk-identity"],
                "seeds": [42],
            }
            with patch.dict(os.environ, {"WM_BENCH_CACHE_ROOT": str(cache_root)}):
                larger = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_prefix_larger",
                        selection={**base_selection, "maxSamples": 4},
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
                )
                smaller = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_prefix_smaller",
                        selection={**base_selection, "maxSamples": 2},
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
                )

            self.assertEqual(larger["status"], "succeeded")
            self.assertEqual(smaller["status"], "succeeded")
            self.assertEqual(smaller["resultUnits"][0]["sampleCount"], 2)
            self.assertNotEqual(
                larger["resultUnits"][0]["materialized"]["watermarkedDir"],
                smaller["resultUnits"][0]["materialized"]["watermarkedDir"],
            )
            smaller_profiles = [
                json.loads(line)
                for line in (runs_root / "run_prefix_smaller" / "runtime_profile.jsonl").read_text().splitlines()
                if line.strip()
            ]
            smaller_reused_stages = {
                record["stage"]
                for record in smaller_profiles
                if record["status"] == "reused"
                and record.get("metadata", {}).get("cacheHit") is True
            }
            self.assertIn("watermark_embed", smaller_reused_stages)
            self.assertIn("attack", smaller_reused_stages)
            self.assertIn("attack_negative_control", smaller_reused_stages)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            cache_root = root / "cache" / "materialized"
            dataset_dir.mkdir(parents=True)
            for index in range(4):
                Image.new("RGB", (96, 96), (80 + index, 120, 180)).save(dataset_dir / f"{index:03d}.png")

            with patch.dict(os.environ, {"WM_BENCH_CACHE_ROOT": str(cache_root)}):
                smaller = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_prefix_seed",
                        selection={**base_selection, "maxSamples": 2},
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
                )
                larger = run_local_experiment(
                    LocalRunRequest(
                        run_id="run_prefix_expand",
                        selection={**base_selection, "maxSamples": 4},
                        resources_root=root / "resources",
                        runs_root=runs_root,
                    )
            )

            self.assertEqual(smaller["status"], "succeeded")
            self.assertEqual(larger["status"], "succeeded")
            self.assertEqual(larger["resultUnits"][0]["sampleCount"], 4)
            larger_profiles = [
                json.loads(line)
                for line in (runs_root / "run_prefix_expand" / "runtime_profile.jsonl").read_text().splitlines()
                if line.strip()
            ]
            partial_fills = [
                record
                for record in larger_profiles
                if record.get("metadata", {}).get("partialFill") is True
            ]
            self.assertGreaterEqual(len(partial_fills), 3)
            self.assertTrue(all(record["metadata"]["reusedSamples"] == 2 for record in partial_fills))
            self.assertTrue(all(record["metadata"]["pendingSamples"] == 2 for record in partial_fills))

    def test_stop_intent_is_written_to_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (90, 140, 210)).save(dataset_dir / "sample.png")

            summary = run_local_experiment(
                LocalRunRequest(
                    run_id="run_pause_requested",
                    selection={
                        "datasetIds": ["smoke"],
                        "algorithmIds": ["alg-traditional-spread-dct"],
                        "attackPresetIds": ["atk-identity"],
                        "seeds": [42],
                        "maxSamples": 1,
                    },
                    resources_root=root / "resources",
                    runs_root=runs_root,
                ),
                should_cancel=lambda: "pause",
            )

            run_root = runs_root / "run_pause_requested"
            run_status = json.loads((run_root / "run_status.json").read_text())
            run_summary = json.loads((run_root / "run_summary.json").read_text())

            self.assertEqual(summary["status"], "paused")
            self.assertEqual(run_status["status"], "paused")
            self.assertEqual(run_summary["status"], "paused")


if __name__ == "__main__":
    unittest.main()
