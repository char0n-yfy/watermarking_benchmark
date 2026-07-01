from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.attacks.base import AttackContext, BaseAttack
from evaluator.attacks.runner import AttackJob, run_attack_dir_with_attack


class DummyBatchAttack(BaseAttack):
    name = "dummy-batch-attack"

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[int] = []

    def apply(self, input_path: Path, output_path: Path, context: AttackContext):
        Image.open(input_path).convert("RGB").save(output_path)
        return {"mode": "single"}

    def apply_batch_impl(self, jobs):
        self.batches.append(len(jobs))
        for input_path, output_path, _context in jobs:
            Image.open(input_path).convert("RGB").save(output_path)
        return [{"mode": "batch"} for _job in jobs]


class ThreadSafeBatchAttack(DummyBatchAttack):
    name = "identity"
    thread_safe_parallel = True


class AttackBatchingTest(unittest.TestCase):
    def _make_inputs(self, root: Path, count: int) -> Path:
        input_dir = root / "input"
        input_dir.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (32, 24), (index * 20, 100, 160)).save(input_dir / f"sample_{index}.png")
        return input_dir

    def test_runner_chunks_attack_batch_impls(self) -> None:
        previous = os.environ.get("WM_BENCH_ATTACK_BATCH_SIZE")
        os.environ["WM_BENCH_ATTACK_BATCH_SIZE"] = "2"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_dir = self._make_inputs(root, 5)
                output_dir = root / "output"
                attack = DummyBatchAttack()
                results = run_attack_dir_with_attack(
                    AttackJob(
                        run_id="run",
                        attack_name=attack.name,
                        params={},
                        input_dir=input_dir,
                        output_dir=output_dir,
                    ),
                    attack,
                )
                self.assertEqual(attack.batches, [2, 2, 1])
                self.assertEqual(len(results), 5)
                self.assertTrue(all(result.ok for result in results))
                self.assertTrue(all(result.metadata["executionMode"] == "batch" for result in results))
                self.assertTrue((output_dir / "attack_manifest.json").exists())
        finally:
            if previous is None:
                os.environ.pop("WM_BENCH_ATTACK_BATCH_SIZE", None)
            else:
                os.environ["WM_BENCH_ATTACK_BATCH_SIZE"] = previous

    def test_threadsafe_attacks_keep_threadpool_priority(self) -> None:
        previous_workers = os.environ.get("WM_BENCH_ATTACK_CPU_WORKERS")
        previous_batch = os.environ.get("WM_BENCH_ATTACK_BATCH_SIZE")
        os.environ["WM_BENCH_ATTACK_CPU_WORKERS"] = "2"
        os.environ["WM_BENCH_ATTACK_BATCH_SIZE"] = "2"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_dir = self._make_inputs(root, 3)
                output_dir = root / "output"
                attack = ThreadSafeBatchAttack()
                results = run_attack_dir_with_attack(
                    AttackJob(
                        run_id="run",
                        attack_name=attack.name,
                        params={},
                        input_dir=input_dir,
                        output_dir=output_dir,
                    ),
                    attack,
                )
                self.assertEqual(attack.batches, [])
                self.assertEqual(len(results), 3)
                self.assertTrue(all(result.metadata["executionMode"] == "threadpool" for result in results))
        finally:
            if previous_workers is None:
                os.environ.pop("WM_BENCH_ATTACK_CPU_WORKERS", None)
            else:
                os.environ["WM_BENCH_ATTACK_CPU_WORKERS"] = previous_workers
            if previous_batch is None:
                os.environ.pop("WM_BENCH_ATTACK_BATCH_SIZE", None)
            else:
                os.environ["WM_BENCH_ATTACK_BATCH_SIZE"] = previous_batch


if __name__ == "__main__":
    unittest.main()
