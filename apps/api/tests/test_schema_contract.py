from __future__ import annotations

import unittest
from pathlib import Path


class SchemaContractTest(unittest.TestCase):
    def test_core_tables_are_declared(self) -> None:
        schema = Path("apps/api/app/db/schema.sql").read_text(encoding="utf-8")
        expected_tables = [
            "users",
            "datasets",
            "dataset_versions",
            "samples",
            "algorithm_packages",
            "algorithm_versions",
            "model_artifacts",
            "attack_methods",
            "attack_presets",
            "experiment_specs",
            "experiment_runs",
            "experiment_cells",
            "worker_heartbeats",
            "artifacts",
            "metric_summaries",
            "sandbox_builds",
        ]

        for table in expected_tables:
            self.assertIn(f"CREATE TABLE {table}", schema)

    def test_status_enums_are_declared(self) -> None:
        schema = Path("apps/api/app/db/schema.sql").read_text(encoding="utf-8")

        for status in ("draft", "queued", "running", "succeeded", "failed", "paused", "cancelled", "partially_failed"):
            self.assertIn(f"'{status}'", schema)


if __name__ == "__main__":
    unittest.main()
