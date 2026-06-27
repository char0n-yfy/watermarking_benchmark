from __future__ import annotations

import unittest
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.planner import AttackSelection, ExperimentSpec, estimate_cell_count, materialize_cells


class PlannerTest(unittest.TestCase):
    def test_materialize_cells_uses_cell_granularity_not_per_image(self) -> None:
        spec = ExperimentSpec(
            spec_id="spec-a",
            dataset_version_ids=("ds-1", "ds-2"),
            algorithm_version_ids=("alg-1", "alg-2"),
            attack_presets=(
                AttackSelection("jpeg-sweep", "jpeg", (0.25, 0.5, 0.75)),
                AttackSelection("identity", "identity", (0.0,)),
            ),
            seeds=(1, 2),
            max_samples_per_dataset=100,
        )

        cells = materialize_cells(spec)

        self.assertEqual(len(cells), 32)
        self.assertEqual(estimate_cell_count(spec), 32)
        self.assertEqual(cells[0].dataset_version_id, "ds-1")
        self.assertEqual(cells[0].attack_method, "jpeg")
        self.assertEqual(cells[-1].seed, 2)

    def test_rejects_empty_dimensions(self) -> None:
        spec = ExperimentSpec(
            spec_id="bad",
            dataset_version_ids=(),
            algorithm_version_ids=("alg-1",),
            attack_presets=(AttackSelection("identity", "identity", (0.0,)),),
            seeds=(42,),
        )

        with self.assertRaises(ValueError):
            materialize_cells(spec)


if __name__ == "__main__":
    unittest.main()
