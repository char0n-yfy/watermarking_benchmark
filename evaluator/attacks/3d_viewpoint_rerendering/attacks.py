from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.attacks.base import AttackContext, BaseAttack
from evaluator.attacks.registry import register_attack


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_ATTACK_WEIGHT_ROOT = PROJECT_ROOT / "resources" / "weights" / "attacks"
DEFAULT_BACKEND_ROOT = PACKAGE_ROOT.parent / "regeneration_attacks" / "backends"
DEFAULT_VIEWPOINT_RERENDERING_ROOT = DEFAULT_ATTACK_WEIGHT_ROOT / "3d_viewpoint_rerendering"
LEGACY_VIEWPOINT_RERENDERING_ROOT = DEFAULT_ATTACK_WEIGHT_ROOT / "regeneration_attacks" / "3d_viewpoint_rerendering"
DEFAULT_SHARP_SOURCE_ROOT = DEFAULT_BACKEND_ROOT / "ml_sharp"
DEFAULT_SHARP_CHECKPOINT_PATH = DEFAULT_VIEWPOINT_RERENDERING_ROOT / "checkpoints" / "sharp_2572gikvuh.pt"
LEGACY_SHARP_CHECKPOINT_PATH = LEGACY_VIEWPOINT_RERENDERING_ROOT / "checkpoints" / "sharp_2572gikvuh.pt"
DEFAULT_SHARP_ATTACK_DEFINITION = "REG-3D-SHARP-Rotate"
DEFAULT_SHARP_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_SHARP_PHASES = tuple(index / 8 for index in range(8))
DEFAULT_SHARP_LOOKAT_MODES = ("point", "ahead")
DEFAULT_MAX_DISPARITY_LEVELS = (0.01, 0.02, 0.04)


def _hash_prefix_from_filename(filename: str) -> str | None:
    match = filename.rsplit("-", 1)
    if len(match) != 2:
        return None
    candidate = match[-1].split(".", 1)[0]
    return candidate if len(candidate) == 8 and all(ch in "0123456789abcdef" for ch in candidate) else None


def _download_file(url: str, target: Path, progress: bool) -> None:
    import torch

    target.parent.mkdir(parents=True, exist_ok=True)
    hash_prefix = _hash_prefix_from_filename(target.name)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        try:
            torch.hub.download_url_to_file(url, str(tmp_path), hash_prefix=hash_prefix, progress=progress)
        except Exception:
            if shutil.which("wget") is None:
                raise
            subprocess.run(
                ["wget", "-q", "-O", str(tmp_path), "--tries=3", "--timeout=60", url],
                check=True,
            )
            if hash_prefix is not None:
                digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
                if not digest.startswith(hash_prefix):
                    raise RuntimeError(
                        f"Downloaded file hash mismatch for {url}: expected prefix {hash_prefix}, got {digest}"
                    )
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _clamp_unit_strength(strength: float) -> float:
    return max(0.0, min(1.0, float(strength)))


def _piecewise_strength_value(strength: float, mild: float, medium: float, strong: float) -> float:
    value = _clamp_unit_strength(strength)
    if value <= 0.5:
        return mild + (medium - mild) * (value / 0.5)
    return medium + (strong - medium) * ((value - 0.5) / 0.5)


def _strength_from_piecewise_value(value: float, mild: float, medium: float, strong: float) -> float:
    value = float(value)
    if value <= medium:
        if medium == mild:
            return 0.0
        return _clamp_unit_strength(0.5 * (value - mild) / (medium - mild))
    if strong == medium:
        return 1.0
    return _clamp_unit_strength(0.5 + 0.5 * (value - medium) / (strong - medium))


def _max_disparity_from_strength(strength: float) -> float:
    mild, medium, strong = DEFAULT_MAX_DISPARITY_LEVELS
    return _piecewise_strength_value(strength, mild, medium, strong)


def _strength_from_max_disparity(max_disparity: float) -> float:
    mild, medium, strong = DEFAULT_MAX_DISPARITY_LEVELS
    return _strength_from_piecewise_value(max_disparity, mild, medium, strong)


def _ensure_sharp_source_root(source_root: str | Path | None) -> Path:
    root = Path(source_root).expanduser() if source_root is not None else DEFAULT_SHARP_SOURCE_ROOT
    required = (
        "src/sharp/cli/predict.py",
        "src/sharp/models/__init__.py",
        "src/sharp/utils/camera.py",
        "src/sharp/utils/gsplat.py",
        "src/sharp/utils/gaussians.py",
    )
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"apple/ml-sharp source is missing under {root}: {missing_list}. "
            f"Place the repository under {DEFAULT_SHARP_SOURCE_ROOT} or pass source_root explicitly."
        )
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return root


def _resolve_sharp_checkpoint(
    checkpoint_path: str | Path | None,
    allow_download: bool,
    progress: bool,
) -> tuple[Path, str, bool]:
    target = Path(checkpoint_path).expanduser() if checkpoint_path is not None else DEFAULT_SHARP_CHECKPOINT_PATH
    if checkpoint_path is None and not target.exists() and LEGACY_SHARP_CHECKPOINT_PATH.exists():
        target = LEGACY_SHARP_CHECKPOINT_PATH
    downloaded = False
    if not target.exists():
        if not allow_download:
            raise FileNotFoundError(f"SHARP checkpoint is missing and download is disabled: {target}")
        _download_file(DEFAULT_SHARP_MODEL_URL, target, progress=progress)
        downloaded = True
    return target, DEFAULT_SHARP_MODEL_URL, downloaded


class ViewpointRerendering3DVariantAttack(BaseAttack):
    name = "3d_viewpoint_rerendering_variant"
    description = "REG-3D-SHARP-Rotate 3D Gaussian viewpoint re-rendering variant."
    phase_index = 0
    phase = 0.0
    lookat_mode = "point"

    def __init__(
        self,
        source_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        strength: float | None = None,
        max_disparity: float | None = None,
        image_size: int | None = None,
        device: str | None = None,
        allow_download: bool = True,
        progress: bool = True,
        save_intermediates: bool = True,
    ) -> None:
        if strength is not None:
            strength = _clamp_unit_strength(float(strength))
            max_disparity = _max_disparity_from_strength(strength)
        elif max_disparity is None:
            max_disparity = DEFAULT_MAX_DISPARITY_LEVELS[1]
            strength = 0.5
        else:
            max_disparity = float(max_disparity)
            strength = _strength_from_max_disparity(max_disparity)
        if max_disparity < 0.0:
            raise ValueError("max_disparity must be non-negative")
        if image_size is not None and int(image_size) <= 0:
            raise ValueError("image_size must be positive when provided")
        if self.lookat_mode not in DEFAULT_SHARP_LOOKAT_MODES:
            valid = ", ".join(DEFAULT_SHARP_LOOKAT_MODES)
            raise ValueError(f"lookat_mode must be one of: {valid}")

        super().__init__(
            source_root=str(source_root) if source_root is not None else str(DEFAULT_SHARP_SOURCE_ROOT),
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else str(DEFAULT_SHARP_CHECKPOINT_PATH),
            attack_definition=DEFAULT_SHARP_ATTACK_DEFINITION,
            strength=float(strength),
            max_disparity=float(max_disparity),
            max_disparity_levels=list(DEFAULT_MAX_DISPARITY_LEVELS),
            trajectory_type="rotate",
            max_zoom=0.0,
            phase_index=int(self.phase_index),
            phase=float(self.phase),
            lookat_mode=self.lookat_mode,
            image_size=None if image_size is None else int(image_size),
            device=device,
            allow_download=bool(allow_download),
            save_intermediates=bool(save_intermediates),
        )
        self.source_root = source_root
        self.checkpoint_path = checkpoint_path
        self.strength = float(strength)
        self.max_disparity = float(max_disparity)
        self.image_size = None if image_size is None else int(image_size)
        self.device_override = device
        self.allow_download = bool(allow_download)
        self.progress = bool(progress)
        self.save_intermediates = bool(save_intermediates)
        self._predictor: Any | None = None
        self._predictor_device: str | None = None
        self._checkpoint_path: Path | None = None
        self._checkpoint_url: str | None = None
        self._checkpoint_downloaded = False
        self._source_root: Path | None = None
        self._sharp_modules: dict[str, Any] = {}

    def _ensure_predictor(self, device: str) -> None:
        import torch

        torch_device = torch.device(device)
        if torch_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("3D Viewpoint Re-rendering with SHARP requires a CUDA GPU for gsplat rendering")
        if self._predictor is not None and self._predictor_device == str(torch_device):
            return

        source_root = _ensure_sharp_source_root(self.source_root)
        checkpoint_path, checkpoint_url, downloaded = _resolve_sharp_checkpoint(
            self.checkpoint_path,
            self.allow_download,
            self.progress,
        )
        try:
            from sharp.cli.predict import predict_image
            from sharp.models import PredictorParams, create_predictor
            from sharp.utils import camera, gsplat, io
            from sharp.utils.gaussians import SceneMetaData, save_ply
        except Exception as exc:
            raise RuntimeError(
                "Unable to import apple/ml-sharp. Install its requirements or place the source tree under "
                f"{DEFAULT_SHARP_SOURCE_ROOT}."
            ) from exc

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval().to(torch_device)

        self._predictor = predictor
        self._predictor_device = str(torch_device)
        self._checkpoint_path = checkpoint_path
        self._checkpoint_url = checkpoint_url
        self._checkpoint_downloaded = downloaded
        self._source_root = source_root
        self._sharp_modules = {
            "predict_image": predict_image,
            "camera": camera,
            "gsplat": gsplat,
            "io": io,
            "SceneMetaData": SceneMetaData,
            "save_ply": save_ply,
        }

    def _variant_output_dir(self, output_path: Path, context: AttackContext) -> Path:
        sample_key = str(context.sample_id or output_path.stem).replace("/", "__").replace(os.sep, "__")
        if context.workspace_dir is not None:
            return context.workspace_dir / "_intermediates" / self.name / sample_key
        return output_path.parent / f"{output_path.stem}_sharp_variant"

    def _render_variant(
        self,
        gaussians: Any,
        metadata: Any,
        *,
        device: str,
        renderer: Any | None = None,
        gaussians_device: Any | None = None,
    ) -> Image.Image:
        import numpy as np
        import torch

        camera = self._sharp_modules["camera"]
        gsplat = self._sharp_modules["gsplat"]
        torch_device = torch.device(device)
        width, height = metadata.resolution_px
        f_px = metadata.focal_length_px
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=torch_device,
            dtype=torch.float32,
        )
        params = camera.TrajectoryParams(
            type="rotate",
            lookat_mode=self.lookat_mode,
            max_disparity=self.max_disparity,
            max_zoom=0.0,
            distance_m=0.0,
            num_steps=1,
            num_repeats=1,
        )
        offset_x, offset_y, _ = camera.compute_max_offset(
            gaussians,
            params,
            resolution_px=metadata.resolution_px,
            f_px=f_px,
        )
        eye_position = torch.tensor(
            [
                float(offset_x) * np.sin(2 * np.pi * self.phase),
                float(offset_y) * np.cos(2 * np.pi * self.phase),
                0.0,
            ],
            dtype=torch.float32,
        )
        camera_model = camera.create_camera_model(
            gaussians,
            intrinsics,
            resolution_px=metadata.resolution_px,
            lookat_mode=self.lookat_mode,
        )
        if renderer is None:
            renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
        if gaussians_device is None:
            gaussians_device = gaussians.to(torch_device)
        camera_info = camera_model.compute(eye_position)
        with torch.inference_mode():
            rendering_output = renderer(
                gaussians_device,
                extrinsics=camera_info.extrinsics[None].to(torch_device),
                intrinsics=camera_info.intrinsics[None].to(torch_device),
                image_width=camera_info.width,
                image_height=camera_info.height,
            )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).clamp(0, 255).to(dtype=torch.uint8)
        return Image.fromarray(color.detach().cpu().numpy(), mode="RGB")

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        import torch

        device = self.device_override or context.device or "cuda"
        self._ensure_predictor(device)
        assert self._predictor is not None
        assert self._checkpoint_path is not None
        assert self._checkpoint_url is not None

        torch_device = torch.device(self._predictor_device or device)
        io = self._sharp_modules["io"]
        predict_image = self._sharp_modules["predict_image"]
        SceneMetaData = self._sharp_modules["SceneMetaData"]
        save_ply = self._sharp_modules["save_ply"]
        gsplat = self._sharp_modules["gsplat"]

        image_np, _, f_px = io.load_rgb(input_path)
        height, width = image_np.shape[:2]
        gaussians = predict_image(self._predictor, image_np, f_px, torch_device)
        metadata = SceneMetaData(float(f_px), (width, height), "linearRGB")

        variant_dir = self._variant_output_dir(output_path, context)
        if self.save_intermediates:
            variant_dir.mkdir(parents=True, exist_ok=True)
            save_ply(gaussians, f_px, (height, width), variant_dir / "scene.ply")

        renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
        rendered = self._render_variant(
            gaussians,
            metadata,
            device=str(torch_device),
            renderer=renderer,
            gaussians_device=gaussians.to(torch_device),
        )
        variant_path = variant_dir / f"phase_{self.phase_index:02d}_{self.lookat_mode}.png"
        if self.save_intermediates:
            rendered.save(variant_path, format="PNG")
        if self.image_size is not None:
            rendered = rendered.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(output_path, format="PNG")
        if str(torch_device).startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "backend": "apple/ml-sharp",
            "attack_definition": DEFAULT_SHARP_ATTACK_DEFINITION,
            "source_root": str(self._source_root),
            "checkpoint_path": str(self._checkpoint_path),
            "checkpoint_url": self._checkpoint_url,
            "checkpoint_downloaded": self._checkpoint_downloaded,
            "trajectory_type": "rotate",
            "strength": self.strength,
            "max_disparity": self.max_disparity,
            "max_disparity_levels": list(DEFAULT_MAX_DISPARITY_LEVELS),
            "max_zoom": 0.0,
            "phase_index": int(self.phase_index),
            "phase": float(self.phase),
            "lookat_mode": self.lookat_mode,
            "variant_count": 1,
            "variant_output_path": str(variant_path) if self.save_intermediates else None,
            "input_size": [width, height],
            "output_size": list(rendered.size),
        }


VIEWPOINT_ATTACK_CLASSES: list[type[ViewpointRerendering3DVariantAttack]] = []


def _register_viewpoint_variant(phase_index: int, phase: float, lookat_mode: str) -> None:
    mode_name = lookat_mode.title().replace("_", "")
    class_name = f"ViewpointRerendering3DPhase{phase_index}{mode_name}Attack"
    method_name = f"3d_viewpoint_rerendering_phase{phase_index}_{lookat_mode}"
    cls = type(
        class_name,
        (ViewpointRerendering3DVariantAttack,),
        {
            "__module__": __name__,
            "name": method_name,
            "description": (
                "REG-3D-SHARP-Rotate 3D Gaussian viewpoint re-rendering "
                f"phase {phase_index}/8 with lookat_mode={lookat_mode}."
            ),
            "phase_index": phase_index,
            "phase": phase,
            "lookat_mode": lookat_mode,
        },
    )
    globals()[class_name] = register_attack(cls)
    VIEWPOINT_ATTACK_CLASSES.append(globals()[class_name])


for _lookat_mode in DEFAULT_SHARP_LOOKAT_MODES:
    for _phase_index, _phase in enumerate(DEFAULT_SHARP_PHASES):
        _register_viewpoint_variant(_phase_index, _phase, _lookat_mode)


__all__ = [
    "DEFAULT_MAX_DISPARITY_LEVELS",
    "DEFAULT_SHARP_LOOKAT_MODES",
    "DEFAULT_SHARP_PHASES",
    "VIEWPOINT_ATTACK_CLASSES",
    "ViewpointRerendering3DVariantAttack",
    *[cls.__name__ for cls in VIEWPOINT_ATTACK_CLASSES],
]
