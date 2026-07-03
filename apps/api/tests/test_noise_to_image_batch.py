from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from evaluator.attacks.base import AttackContext
from evaluator.attacks.regeneration_attacks.attacks import NoiseToImageAttack


class _FakeCtrlRegenPipe:
    def __init__(self) -> None:
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.last_kwargs: dict[str, object] | None = None

    def __call__(self, *args: object, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = dict(kwargs)
        images = kwargs["image"]
        control_image = kwargs["control_image"]
        if not isinstance(images, list):
            raise AssertionError("image must be a batch list")
        if not isinstance(control_image, list) or len(control_image) != 1:
            raise ValueError("ControlNet batch input length mismatch")
        control_batch = control_image[0]
        if not isinstance(control_batch, list) or len(control_batch) != len(images):
            raise ValueError("ControlNet batch input length mismatch")
        return SimpleNamespace(images=[image.copy() for image in images])


class NoiseToImageBatchTest(unittest.TestCase):
    def test_batch_wraps_control_images_for_single_controlnet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_paths: list[Path] = []
            jobs = []
            for index in range(2):
                input_path = root / f"in_{index}.png"
                output_path = root / f"out_{index}.png"
                Image.new("RGB", (32, 24), (40 + index, 80, 120)).save(input_path)
                input_paths.append(input_path)
                jobs.append(
                    (
                        input_path,
                        output_path,
                        AttackContext(
                            run_id="run",
                            sample_id=f"sample_{index}",
                            attack_name="noise_to_image",
                            params={},
                            workspace_dir=root,
                            device="cpu",
                            seed=2026 + index,
                        ),
                    )
                )

            attack = NoiseToImageAttack(image_size=32, num_inference_steps=1, step=0.5)
            fake_pipe = _FakeCtrlRegenPipe()

            def fake_ensure_pipe(device: str) -> None:
                attack._pipe = fake_pipe
                attack._pipe_device = "cpu"
                attack._canny_impl = lambda image: image.copy()
                attack._color_match = lambda _ref, image: image
                attack._resolved_paths = {}

            attack._ensure_pipe = fake_ensure_pipe  # type: ignore[method-assign]

            metadatas = attack.apply_batch_impl(jobs)

            self.assertEqual(len(metadatas), 2)
            self.assertTrue(all(output_path.exists() for _input_path, output_path, _context in jobs))
            self.assertIsNotNone(fake_pipe.last_kwargs)
            control_image = fake_pipe.last_kwargs["control_image"] if fake_pipe.last_kwargs else None
            self.assertIsInstance(control_image, list)
            self.assertEqual(len(control_image), 1)
            self.assertEqual(len(control_image[0]), 2)


if __name__ == "__main__":
    unittest.main()
