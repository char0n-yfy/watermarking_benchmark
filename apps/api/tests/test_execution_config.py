from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.execution import resolve_named_cpu_workers


class ExecutionConfigTest(unittest.TestCase):
    def test_named_cpu_worker_override_precedes_global_value(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WM_BENCH_WATERMARK_CPU_WORKERS": "8",
                "WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD": "foo=3,bar=12",
            },
            clear=False,
        ):
            resolved = resolve_named_cpu_workers(
                "foo",
                overrides_env="WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD",
                global_env="WM_BENCH_WATERMARK_CPU_WORKERS",
                job_count=10,
            )
            capped = resolve_named_cpu_workers(
                "bar",
                overrides_env="WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD",
                global_env="WM_BENCH_WATERMARK_CPU_WORKERS",
                job_count=5,
            )
            fallback = resolve_named_cpu_workers(
                "baz",
                overrides_env="WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD",
                global_env="WM_BENCH_WATERMARK_CPU_WORKERS",
                job_count=10,
            )

        self.assertEqual(resolved.value, 3)
        self.assertEqual(
            resolved.source,
            "env:WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD:foo",
        )
        self.assertEqual(capped.value, 5)
        self.assertEqual(fallback.value, 8)
        self.assertEqual(fallback.source, "env:WM_BENCH_WATERMARK_CPU_WORKERS")

    def test_named_cpu_worker_respects_disabled_and_single_job(self) -> None:
        with patch.dict(
            os.environ,
            {"WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD": "brightness=8"},
            clear=False,
        ):
            disabled = resolve_named_cpu_workers(
                "brightness",
                overrides_env="WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD",
                global_env="WM_BENCH_ATTACK_CPU_WORKERS",
                job_count=12,
                enabled=False,
            )
            single = resolve_named_cpu_workers(
                "brightness",
                overrides_env="WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD",
                global_env="WM_BENCH_ATTACK_CPU_WORKERS",
                job_count=1,
            )

        self.assertEqual(disabled.value, 1)
        self.assertEqual(disabled.source, "disabled")
        self.assertEqual(single.value, 1)
        self.assertEqual(single.source, "single_job")


if __name__ == "__main__":
    unittest.main()
