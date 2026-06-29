from __future__ import annotations

import re
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps

from evaluator.attacks.base import AttackContext, BaseAttack
from evaluator.attacks.registry import register_attack


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_WEIGHT_ROOT = PROJECT_ROOT / "resources" / "weights" / "attacks" / "regeneration_attacks"
DEFAULT_DIFFUSION_MODEL_ROOT = DEFAULT_WEIGHT_ROOT / "diffusion" / "sd2-1-base"
DEFAULT_NOISE_TO_IMAGE_ROOT = DEFAULT_WEIGHT_ROOT / "noise_to_image"
DEFAULT_BACKEND_ROOT = PACKAGE_ROOT / "backends"
DEFAULT_NOISE_TO_IMAGE_SOURCE_ROOT = DEFAULT_BACKEND_ROOT / "ctrlregen"
DEFAULT_IMAGE_TO_VEDIO_SOURCE_ROOT = DEFAULT_BACKEND_ROOT / "nfpa"
DEFAULT_DIFFUSION_REPO_ID = "sd2-community/stable-diffusion-2-1-base"
DEFAULT_CTRLREGEN_BASE_REPO_ID = "SG161222/Realistic_Vision_V4.0_noVAE"
DEFAULT_CTRLREGEN_ADAPTER_REPO_ID = "yepengliu/ctrlregen"
DEFAULT_CTRLREGEN_IMAGE_ENCODER_REPO_ID = "facebook/dinov2-giant"
DEFAULT_CTRLREGEN_VAE_REPO_ID = "stabilityai/sd-vae-ft-mse"
DEFAULT_REGEN_NOISE_STEP_RANGE = (20, 100)

DIFFUSION_REQUIRED_FILES = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

DIFFUSION_ALLOW_PATTERNS = (
    "README.md",
    *DIFFUSION_REQUIRED_FILES,
)

SUPPORTED_VAE_MODELS = (
    "bmshj2018-factorized",
    "cheng2020-anchor",
    "bmshj2018-hyperprior",
    "mbt2018-mean",
)

COMPRESSAI_ROOT_URL = "https://compressai.s3.amazonaws.com/models/v1"
VAE_WEIGHT_URLS: dict[str, dict[int, str]] = {
    "bmshj2018-factorized": {
        1: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
        2: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-2-87279a02.pth.tar",
        3: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-3-5c6f152b.pth.tar",
        4: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-4-1ed4405a.pth.tar",
        5: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-5-866ba797.pth.tar",
        6: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-6-9b02ea3a.pth.tar",
        7: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-7-6dfd6734.pth.tar",
        8: f"{COMPRESSAI_ROOT_URL}/bmshj2018-factorized-prior-8-5232faa3.pth.tar",
    },
    "cheng2020-anchor": {
        1: f"{COMPRESSAI_ROOT_URL}/cheng2020-anchor-1-dad2ebff.pth.tar",
        2: f"{COMPRESSAI_ROOT_URL}/cheng2020-anchor-2-a29008eb.pth.tar",
        3: f"{COMPRESSAI_ROOT_URL}/cheng2020-anchor-3-e49be189.pth.tar",
        4: f"{COMPRESSAI_ROOT_URL}/cheng2020-anchor-4-98b0b468.pth.tar",
        5: f"{COMPRESSAI_ROOT_URL}/cheng2020-anchor-5-23852949.pth.tar",
        6: f"{COMPRESSAI_ROOT_URL}/cheng2020-anchor-6-4c052b1a.pth.tar",
    },
    "bmshj2018-hyperprior": {
        1: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-1-7eb97409.pth.tar",
        2: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-2-93677231.pth.tar",
        3: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-3-6d87be32.pth.tar",
        4: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-4-de1b779c.pth.tar",
        5: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-5-f8b614e1.pth.tar",
        6: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
        7: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-7-3804dcbd.pth.tar",
        8: f"{COMPRESSAI_ROOT_URL}/bmshj2018-hyperprior-8-a583f0cf.pth.tar",
    },
    "mbt2018-mean": {
        1: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-1-e522738d.pth.tar",
        2: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-2-e54a039d.pth.tar",
        3: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-3-723404a8.pth.tar",
        4: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-4-6dba02a3.pth.tar",
        5: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-5-d504e8eb.pth.tar",
        6: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-6-a19628ab.pth.tar",
        7: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-7-d5d441d1.pth.tar",
        8: f"{COMPRESSAI_ROOT_URL}/mbt2018-mean-8-8089ae3e.pth.tar",
    },
}


def _validate_vae_choice(model_name: str, quality: int) -> None:
    if model_name not in SUPPORTED_VAE_MODELS:
        choices = ", ".join(SUPPORTED_VAE_MODELS)
        raise ValueError(f"Unsupported VAE model '{model_name}'. Choose one of: {choices}")
    if quality not in VAE_WEIGHT_URLS[model_name]:
        valid = ", ".join(str(item) for item in sorted(VAE_WEIGHT_URLS[model_name]))
        raise ValueError(f"Unsupported quality {quality} for {model_name}. Valid qualities: {valid}")


def _filename_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def _hash_prefix_from_filename(filename: str) -> str | None:
    match = re.search(r"-([0-9a-f]{8})\.pth\.tar$", filename)
    return match.group(1) if match else None


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


def _resolve_vae_weight(
    model_name: str,
    quality: int,
    weight_root: str | Path | None,
    allow_download: bool,
    progress: bool,
) -> tuple[Path, str, bool]:
    _validate_vae_choice(model_name, quality)
    root = DEFAULT_WEIGHT_ROOT if weight_root is None else Path(weight_root).expanduser()
    model_root = root / "vae" / model_name
    url = VAE_WEIGHT_URLS[model_name][quality]
    filename = _filename_from_url(url)
    weight_path = model_root / filename

    downloaded = False
    if not weight_path.exists():
        if not allow_download:
            raise FileNotFoundError(f"VAE weight is missing and download is disabled: {weight_path}")
        _download_file(url, weight_path, progress=progress)
        downloaded = True
    return weight_path, url, downloaded


def _load_compressai_model(model_name: str, quality: int, metric: str, weight_path: Path, device: str):
    import torch

    try:
        from compressai.zoo.image import model_architectures
        from compressai.zoo.pretrained import load_pretrained
    except Exception as exc:
        raise RuntimeError(
            "Unable to import CompressAI. Check that compressai, scipy, and numpy are ABI-compatible "
            "in the active Python environment."
        ) from exc

    if metric != "mse":
        raise ValueError("Local regeneration VAE weights currently support metric='mse' only")

    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = load_pretrained(state_dict)
    model = model_architectures[model_name].from_state_dict(state_dict)
    model.eval().to(device)
    return model


def _prepare_image(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def _square_image(image: Image.Image, size: int, mode: str = "fit") -> Image.Image:
    mode_l = (mode or "fit").lower()
    if mode_l == "pad":
        return ImageOps.pad(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)


def _load_module_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required backend source file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_backend_source_root(
    source_root: str | Path | None,
    default_root: Path,
    required_files: tuple[str, ...],
    backend_name: str,
) -> Path:
    root = Path(source_root).expanduser() if source_root is not None else default_root
    missing = [relative for relative in required_files if not (root / relative).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"{backend_name} backend source is missing under {root}: {missing_list}. "
            f"Place the official source tree under {default_root} or pass source_root explicitly."
        )
    return root


def _has_any_file(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(child.is_file() for child in path.rglob("*"))


def _is_local_reference(value: str) -> bool:
    path = Path(value).expanduser()
    return (
        path.is_absolute()
        or value.startswith("./")
        or value.startswith("../")
        or value.startswith("~/")
    )


def _download_hf_snapshot(
    repo_id: str,
    local_dir: Path,
    allow_download: bool,
    allow_patterns: tuple[str, ...] | None = None,
) -> tuple[Path, bool]:
    if _has_any_file(local_dir):
        return local_dir, False
    if not allow_download:
        raise FileNotFoundError(f"Hugging Face snapshot is missing and download is disabled: {local_dir}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required to download {repo_id}") from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
    )
    return local_dir, True


def _resolve_hf_or_path(
    value: str | Path | None,
    *,
    default_repo_id: str,
    local_dir: Path,
    allow_download: bool,
    allow_patterns: tuple[str, ...] | None = None,
) -> tuple[str, bool, str]:
    raw = default_repo_id if value is None else str(value)
    path = Path(raw).expanduser()
    if path.exists():
        return str(path.resolve()), False, raw
    if _is_local_reference(raw):
        raise FileNotFoundError(f"Configured local model path does not exist: {path}")
    resolved, downloaded = _download_hf_snapshot(raw, local_dir, allow_download, allow_patterns)
    return str(resolved.resolve()), downloaded, raw


def _ensure_torch_diffusers_env() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import transformers  # type: ignore
        from transformers import utils as transformers_utils  # type: ignore

        if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
            transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
        if not hasattr(transformers, "FLAX_WEIGHTS_NAME"):
            transformers.FLAX_WEIGHTS_NAME = transformers_utils.FLAX_WEIGHTS_NAME
        if not hasattr(transformers, "CLIPFeatureExtractor") and hasattr(transformers, "CLIPImageProcessor"):
            transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor
    except Exception:
        pass


def _generator_device(device: Any) -> str:
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    return "cpu" if device_type == "mps" else device_type


def _torch_dtype_from_name(dtype_name: str, device: str):
    import torch

    normalized = dtype_name.lower()
    if normalized == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "full"}:
        return torch.float32
    raise ValueError("dtype must be one of: auto, float16, bfloat16, float32")


def _resolve_diffusion_model_root(
    model_root: str | Path | None,
    weight_root: str | Path | None,
    allow_download: bool,
) -> tuple[Path, bool, str]:
    root = Path(model_root).expanduser() if model_root is not None else None
    if root is None:
        weight_base = DEFAULT_WEIGHT_ROOT if weight_root is None else Path(weight_root).expanduser()
        root = weight_base / "diffusion" / "sd2-1-base"
    missing = [relative for relative in DIFFUSION_REQUIRED_FILES if not (root / relative).exists()]
    downloaded = False
    if missing:
        if not allow_download:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(f"Stable Diffusion 2.1-base files are missing under {root}: {missing_list}")
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError("huggingface_hub is required to download Stable Diffusion 2.1-base weights") from exc
        snapshot_download(
            repo_id=DEFAULT_DIFFUSION_REPO_ID,
            local_dir=str(root),
            allow_patterns=list(DIFFUSION_ALLOW_PATTERNS),
        )
        downloaded = True
        missing = [relative for relative in DIFFUSION_REQUIRED_FILES if not (root / relative).exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(f"Stable Diffusion 2.1-base download did not produce: {missing_list}")
    return root, downloaded, DEFAULT_DIFFUSION_REPO_ID


def _load_resd_pipeline(
    model_root: Path,
    device: str,
    dtype_name: str,
    local_files_only: bool,
):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    try:
        from diffusers import StableDiffusionPipeline
    except Exception as exc:
        raise RuntimeError(
            "Unable to import Diffusers. In this environment, set USE_TF=0 before importing Diffusers "
            "to keep transformers from probing TensorFlow/Keras."
        ) from exc

    torch_dtype = _torch_dtype_from_name(dtype_name, device)
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_root),
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        use_safetensors=True,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe, torch_dtype


def _prompt_embeddings(pipe, prompt: str, negative_prompt: str | None, device: str, do_cfg: bool):
    if hasattr(pipe, "encode_prompt"):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
        )[:2]
        if do_cfg:
            return torch_cat_negative_positive(negative_prompt_embeds, prompt_embeds)
        return prompt_embeds
    return pipe._encode_prompt(prompt, device, 1, do_cfg, negative_prompt)


def torch_cat_negative_positive(negative_prompt_embeds: Any, prompt_embeds: Any):
    import torch

    return torch.cat([negative_prompt_embeds, prompt_embeds])


def _head_start_step(noise_step: int, num_inference_steps: int, num_train_timesteps: int, override: int | None) -> int:
    if override is not None:
        return max(0, min(int(override), num_inference_steps - 1))
    step_stride = max(num_train_timesteps // num_inference_steps, 1)
    reverse_steps = max(noise_step // step_stride, 1)
    return max(0, min(num_inference_steps - reverse_steps, num_inference_steps - 1))


def _image_to_latents(pipe, image: Image.Image, generator: Any, device: str):
    import numpy as np
    import torch

    vae_dtype = getattr(pipe.vae, "dtype", torch.float32)
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    image_np = image_np * 2.0 - 1.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=vae_dtype)
    latent_dist = pipe.vae.encode(tensor).latent_dist
    try:
        latents = latent_dist.sample(generator=generator)
    except TypeError:
        latents = latent_dist.sample(generator)
    return latents * pipe.vae.config.scaling_factor


def _latents_to_image(pipe, latents: Any) -> Image.Image:
    import numpy as np
    import torch

    scaling_factor = pipe.vae.config.scaling_factor
    with torch.no_grad():
        decoded = pipe.vae.decode(latents / scaling_factor, return_dict=False)[0]
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
    image_np = decoded.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image_uint8 = (image_np * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_uint8)


def _run_diffusion_regeneration(
    pipe: Any,
    image: Image.Image,
    *,
    prompt: str,
    negative_prompt: str | None,
    noise_step: int,
    num_inference_steps: int,
    guidance_scale: float,
    eta: float,
    seed: int,
    device: str,
    head_start_step: int | None,
) -> tuple[Image.Image, Mapping[str, Any]]:
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    if noise_step < 0:
        raise ValueError("noise_step must be non-negative")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")

    torch_device = torch.device(device)
    generator_device = "cpu" if torch_device.type == "mps" else torch_device
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    num_train_timesteps = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
    if noise_step >= num_train_timesteps:
        raise ValueError(f"noise_step must be < scheduler num_train_timesteps ({num_train_timesteps})")

    with torch.inference_mode():
        latents = _image_to_latents(pipe, image, generator, torch_device)
        noise = randn_tensor(latents.shape, generator=generator, device=torch_device, dtype=latents.dtype)
        timestep = torch.tensor([noise_step], dtype=torch.long, device=torch_device)
        latents = pipe.scheduler.add_noise(latents, noise, timestep)

        do_cfg = guidance_scale > 1.0
        prompt_embeds = _prompt_embeddings(pipe, prompt, negative_prompt, torch_device, do_cfg)
        pipe.scheduler.set_timesteps(num_inference_steps, device=torch_device)
        timesteps = pipe.scheduler.timesteps
        start_step = _head_start_step(noise_step, num_inference_steps, num_train_timesteps, head_start_step)
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        for index, timestep_value in enumerate(timesteps):
            if index < start_step:
                continue
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep_value)
            noise_pred = pipe.unet(latent_model_input, timestep_value, encoder_hidden_states=prompt_embeds).sample
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = pipe.scheduler.step(noise_pred, timestep_value, latents, **extra_step_kwargs).prev_sample

        image_out = _latents_to_image(pipe, latents)

    return image_out, {
        "seed": seed,
        "noise_step": noise_step,
        "num_inference_steps": num_inference_steps,
        "head_start_step": start_step,
        "guidance_scale": guidance_scale,
        "eta": eta,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_train_timesteps": num_train_timesteps,
        "scheduler": pipe.scheduler.__class__.__name__,
    }


def _clamp_unit_strength(strength: float) -> float:
    return max(0.0, min(1.0, float(strength)))


def _noise_step_from_strength(strength: float, noise_step_min: int, noise_step_max: int) -> int:
    if noise_step_min < 0:
        raise ValueError("noise_step_min must be non-negative")
    if noise_step_max < noise_step_min:
        raise ValueError("noise_step_max must be >= noise_step_min")
    return int(round(noise_step_min + (noise_step_max - noise_step_min) * _clamp_unit_strength(strength)))


def _strength_from_noise_step(noise_step: int, noise_step_min: int, noise_step_max: int) -> float:
    if noise_step_max == noise_step_min:
        return 0.0
    return _clamp_unit_strength((int(noise_step) - noise_step_min) / (noise_step_max - noise_step_min))


@register_attack
class RegenVAEAttack(BaseAttack):
    name = "regen_vae"
    description = "Regenerate an image through a configurable CompressAI VAE reconstruction model."

    def __init__(
        self,
        vae_model_name: str = "cheng2020-anchor",
        model_name: str | None = None,
        quality: int = 3,
        metric: str = "mse",
        image_size: int = 512,
        weight_root: str | Path | None = None,
        allow_download: bool = True,
        progress: bool = True,
    ) -> None:
        if model_name is not None:
            vae_model_name = model_name
        quality = int(quality)
        image_size = int(image_size)
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        _validate_vae_choice(vae_model_name, quality)
        super().__init__(
            vae_model_name=vae_model_name,
            quality=quality,
            metric=metric,
            image_size=image_size,
            weight_root=str(weight_root) if weight_root is not None else str(DEFAULT_WEIGHT_ROOT),
            allow_download=bool(allow_download),
        )
        self.vae_model_name = vae_model_name
        self.quality = quality
        self.metric = metric
        self.image_size = image_size
        self.weight_root = weight_root
        self.allow_download = bool(allow_download)
        self.progress = bool(progress)
        self._model: Any | None = None
        self._model_device: str | None = None
        self._weight_path: Path | None = None
        self._weight_url: str | None = None
        self._downloaded = False

    def _ensure_model(self, device: str) -> None:
        if self._model is not None and self._model_device == device:
            return
        weight_path, url, downloaded = _resolve_vae_weight(
            self.vae_model_name,
            self.quality,
            self.weight_root,
            self.allow_download,
            self.progress,
        )
        self._model = _load_compressai_model(self.vae_model_name, self.quality, self.metric, weight_path, device)
        self._model_device = device
        self._weight_path = weight_path
        self._weight_url = url
        self._downloaded = downloaded

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        import torch
        from torchvision import transforms

        device = context.device or "cpu"
        self._ensure_model(device)
        assert self._model is not None
        assert self._weight_path is not None
        assert self._weight_url is not None

        image = Image.open(input_path)
        input_size = image.size
        prepared = _prepare_image(image, self.image_size)
        tensor = transforms.ToTensor()(prepared).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self._model(tensor)
            tensor_out = output["x_hat"].clamp(0, 1)

        attacked = transforms.ToPILImage()(tensor_out.squeeze(0).cpu())
        attacked.save(output_path)
        return {
            "backend": "compressai",
            "vae_model_name": self.vae_model_name,
            "quality": self.quality,
            "metric": self.metric,
            "image_size": self.image_size,
            "input_size": list(input_size),
            "output_size": list(attacked.size),
            "weight_path": str(self._weight_path),
            "weight_url": self._weight_url,
            "weight_downloaded": self._downloaded,
            "diffusion_model_root": str(DEFAULT_DIFFUSION_MODEL_ROOT),
        }


class _BaseRegenDiffusionAttack(BaseAttack):
    description = "Regenerate an image through Stable Diffusion latent noising and denoising."
    passes = 1

    def __init__(
        self,
        noise_step: int | None = None,
        strength: float | None = None,
        noise_step_min: int = DEFAULT_REGEN_NOISE_STEP_RANGE[0],
        noise_step_max: int = DEFAULT_REGEN_NOISE_STEP_RANGE[1],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_size: int = 512,
        prompt: str = "",
        negative_prompt: str | None = None,
        seed: int | None = 1024,
        eta: float = 0.0,
        head_start_step: int | None = None,
        model_root: str | Path | None = None,
        weight_root: str | Path | None = None,
        allow_download: bool = True,
        local_files_only: bool = True,
        dtype: str = "auto",
        save_intermediates: bool = False,
    ) -> None:
        noise_step_min = int(noise_step_min)
        noise_step_max = int(noise_step_max)
        if noise_step_min < 0:
            raise ValueError("noise_step_min must be non-negative")
        if noise_step_max < noise_step_min:
            raise ValueError("noise_step_max must be >= noise_step_min")
        if strength is not None:
            strength = _clamp_unit_strength(float(strength))
            noise_step = _noise_step_from_strength(strength, noise_step_min, noise_step_max)
        elif noise_step is None:
            noise_step = 60
            strength = _strength_from_noise_step(noise_step, noise_step_min, noise_step_max)
        else:
            noise_step = int(noise_step)
            strength = _strength_from_noise_step(noise_step, noise_step_min, noise_step_max)
        num_inference_steps = int(num_inference_steps)
        image_size = int(image_size)
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if noise_step < 0:
            raise ValueError("noise_step must be non-negative")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if head_start_step is not None:
            head_start_step = int(head_start_step)
        super().__init__(
            noise_step=noise_step,
            strength=strength,
            noise_step_min=noise_step_min,
            noise_step_max=noise_step_max,
            num_inference_steps=num_inference_steps,
            guidance_scale=float(guidance_scale),
            image_size=image_size,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            eta=float(eta),
            head_start_step=head_start_step,
            model_root=str(model_root) if model_root is not None else str(DEFAULT_DIFFUSION_MODEL_ROOT),
            weight_root=str(weight_root) if weight_root is not None else str(DEFAULT_WEIGHT_ROOT),
            allow_download=bool(allow_download),
            local_files_only=bool(local_files_only),
            dtype=dtype,
            save_intermediates=bool(save_intermediates),
            passes=self.passes,
        )
        self.noise_step = noise_step
        self.strength = float(strength)
        self.noise_step_min = noise_step_min
        self.noise_step_max = noise_step_max
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = float(guidance_scale)
        self.image_size = image_size
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.eta = float(eta)
        self.head_start_step = head_start_step
        self.model_root = model_root
        self.weight_root = weight_root
        self.allow_download = bool(allow_download)
        self.local_files_only = bool(local_files_only)
        self.dtype = dtype
        self.save_intermediates = bool(save_intermediates)
        self._pipe: Any | None = None
        self._pipe_device: str | None = None
        self._model_path: Path | None = None
        self._model_downloaded = False
        self._model_repo_id: str | None = None
        self._torch_dtype: Any | None = None

    def _ensure_pipe(self, device: str) -> None:
        if self._pipe is not None and self._pipe_device == device:
            return
        model_path, downloaded, repo_id = _resolve_diffusion_model_root(
            self.model_root,
            self.weight_root,
            self.allow_download,
        )
        pipe, torch_dtype = _load_resd_pipeline(
            model_path,
            device,
            self.dtype,
            local_files_only=self.local_files_only and not downloaded,
        )
        self._pipe = pipe
        self._pipe_device = device
        self._model_path = model_path
        self._model_downloaded = downloaded
        self._model_repo_id = repo_id
        self._torch_dtype = torch_dtype

    def _seed_for_context(self, context: AttackContext) -> int:
        if self.seed is not None:
            return int(self.seed)
        if context.seed is not None:
            return int(context.seed)
        return 1024

    def _intermediate_path(self, context: AttackContext, pass_index: int) -> Path | None:
        if not self.save_intermediates or context.workspace_dir is None:
            return None
        relative = Path(context.sample_id)
        return context.workspace_dir / "_intermediates" / self.name / relative.parent / f"{relative.name}.pass{pass_index}.png"

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        import torch

        device = context.device or "cpu"
        self._ensure_pipe(device)
        assert self._pipe is not None
        assert self._model_path is not None
        assert self._model_repo_id is not None

        base_seed = self._seed_for_context(context)
        image = Image.open(input_path)
        input_size = image.size
        current = _prepare_image(image, self.image_size)
        pass_metadata: list[Mapping[str, Any]] = []
        intermediate_paths: list[str] = []

        for pass_index in range(1, self.passes + 1):
            current, metadata = _run_diffusion_regeneration(
                self._pipe,
                current,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                noise_step=self.noise_step,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=self.eta,
                seed=base_seed,
                device=device,
                head_start_step=self.head_start_step,
            )
            metadata = dict(metadata)
            metadata["pass_index"] = pass_index
            pass_metadata.append(metadata)
            intermediate_path = self._intermediate_path(context, pass_index)
            if intermediate_path is not None:
                intermediate_path.parent.mkdir(parents=True, exist_ok=True)
                current.save(intermediate_path)
                intermediate_paths.append(str(intermediate_path))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        current.save(output_path)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {
            "backend": "diffusers",
            "model_name": "StableDiffusion2.1-base",
            "model_repo_id": self._model_repo_id,
            "model_path": str(self._model_path),
            "model_downloaded": self._model_downloaded,
            "dtype": str(self._torch_dtype).replace("torch.", ""),
            "passes": self.passes,
            "strength": self.strength,
            "noise_step": self.noise_step,
            "noise_step_min": self.noise_step_min,
            "noise_step_max": self.noise_step_max,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "image_size": self.image_size,
            "input_size": list(input_size),
            "output_size": list(current.size),
            "pass_metadata": pass_metadata,
            "intermediate_paths": intermediate_paths,
        }


@register_attack
class RegenDiffusionAttack(_BaseRegenDiffusionAttack):
    name = "regen_diffusion"
    description = "Single-pass Stable Diffusion 2.1-base regeneration attack."
    passes = 1


@register_attack
class TwoTimesRegenDiffusionAttack(_BaseRegenDiffusionAttack):
    name = "2x_regen"
    description = "Two repeated Stable Diffusion 2.1-base regeneration passes."
    passes = 2


@register_attack
class FourTimesRegenDiffusionAttack(_BaseRegenDiffusionAttack):
    name = "4x_regen"
    description = "Four repeated Stable Diffusion 2.1-base regeneration passes."
    passes = 4


@register_attack
class NoiseToImageAttack(BaseAttack):
    name = "noise_to_image"
    description = "CtrlRegen noise-to-image controllable regeneration attack."

    def __init__(
        self,
        source_root: str | Path | None = None,
        weight_root: str | Path | None = None,
        base_model: str | Path | None = None,
        spatial_control_path: str | Path | None = None,
        semantic_control_path: str | Path | None = None,
        semantic_control_name: str = "semantic_control_ckp_435000.bin",
        image_encoder: str | Path | None = None,
        vae_model: str | Path | None = None,
        image_size: int = 512,
        square_mode: str = "fit",
        step: float = 1.0,
        seed: int | None = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        controlnet_conditioning_scale: float = 1.0,
        low_threshold: int = 100,
        high_threshold: int = 150,
        prompt: str = "best quality, high quality",
        negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality",
        allow_download: bool = True,
        dtype: str = "auto",
    ) -> None:
        image_size = int(image_size)
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        step = max(0.0, min(1.0, float(step)))
        super().__init__(
            source_root=str(source_root) if source_root is not None else str(DEFAULT_NOISE_TO_IMAGE_SOURCE_ROOT),
            weight_root=str(weight_root) if weight_root is not None else str(DEFAULT_WEIGHT_ROOT),
            base_model=str(base_model) if base_model is not None else DEFAULT_CTRLREGEN_BASE_REPO_ID,
            spatial_control_path=str(spatial_control_path) if spatial_control_path is not None else None,
            semantic_control_path=str(semantic_control_path) if semantic_control_path is not None else None,
            semantic_control_name=semantic_control_name,
            image_encoder=str(image_encoder) if image_encoder is not None else DEFAULT_CTRLREGEN_IMAGE_ENCODER_REPO_ID,
            vae_model=str(vae_model) if vae_model is not None else DEFAULT_CTRLREGEN_VAE_REPO_ID,
            image_size=image_size,
            square_mode=square_mode,
            step=step,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            low_threshold=int(low_threshold),
            high_threshold=int(high_threshold),
            prompt=prompt,
            negative_prompt=negative_prompt,
            allow_download=bool(allow_download),
            dtype=dtype,
        )
        self.source_root = source_root
        self.weight_root = weight_root
        self.base_model = base_model
        self.spatial_control_path = spatial_control_path
        self.semantic_control_path = semantic_control_path
        self.semantic_control_name = semantic_control_name
        self.image_encoder = image_encoder
        self.vae_model = vae_model
        self.image_size = image_size
        self.square_mode = square_mode
        self.step = step
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = float(guidance_scale)
        self.controlnet_conditioning_scale = float(controlnet_conditioning_scale)
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.allow_download = bool(allow_download)
        self.dtype = dtype
        self._pipe: Any | None = None
        self._pipe_device: str | None = None
        self._canny_impl: Any | None = None
        self._color_match: Any | None = None
        self._resolved_paths: dict[str, Any] = {}

    def _seed_for_context(self, context: AttackContext) -> int:
        if self.seed is not None:
            return int(self.seed)
        if context.seed is not None:
            return int(context.seed)
        return 1

    def _resolve_ctrlregen_assets(self) -> dict[str, Any]:
        weight_root = DEFAULT_WEIGHT_ROOT if self.weight_root is None else Path(self.weight_root).expanduser()
        backend_root = weight_root / "noise_to_image"
        source_root = _ensure_backend_source_root(
            self.source_root,
            DEFAULT_NOISE_TO_IMAGE_SOURCE_ROOT,
            ("custom_ip_adapter.py", "custom_i2i_pipeline.py"),
            "CtrlRegen",
        )
        base_ref, base_downloaded, base_repo = _resolve_hf_or_path(
            self.base_model,
            default_repo_id=DEFAULT_CTRLREGEN_BASE_REPO_ID,
            local_dir=backend_root / "base_model" / "Realistic_Vision_V4.0_noVAE",
            allow_download=self.allow_download,
        )
        encoder_ref, encoder_downloaded, encoder_repo = _resolve_hf_or_path(
            self.image_encoder,
            default_repo_id=DEFAULT_CTRLREGEN_IMAGE_ENCODER_REPO_ID,
            local_dir=backend_root / "image_encoder" / "dinov2-giant",
            allow_download=self.allow_download,
        )
        vae_ref, vae_downloaded, vae_repo = _resolve_hf_or_path(
            self.vae_model,
            default_repo_id=DEFAULT_CTRLREGEN_VAE_REPO_ID,
            local_dir=backend_root / "vae" / "sd-vae-ft-mse",
            allow_download=self.allow_download,
        )

        adapter_root = backend_root / "adapters" / "ctrlregen"
        default_spatial = adapter_root / "spatialnet_ckp" / "spatial_control_ckp_14000"
        default_semantic = adapter_root / "semanticnet_ckp"
        semantic_weight = default_semantic / "models" / self.semantic_control_name
        adapter_downloaded = False
        if self.spatial_control_path is None and self.semantic_control_path is None:
            if not default_spatial.exists() or not semantic_weight.exists():
                _, adapter_downloaded = _download_hf_snapshot(
                    DEFAULT_CTRLREGEN_ADAPTER_REPO_ID,
                    adapter_root,
                    self.allow_download,
                    allow_patterns=(
                        "README.md",
                        "spatialnet_ckp/**",
                        "semanticnet_ckp/**",
                    ),
                )
            spatial_ref = str(default_spatial.resolve())
            spatial_subfolder = None
            spatial_local_only = True
            semantic_ref = str(default_semantic.resolve())
            semantic_subfolder = "models"
            semantic_local_only = True
        else:
            spatial_path = Path(self.spatial_control_path).expanduser() if self.spatial_control_path is not None else default_spatial
            semantic_path = Path(self.semantic_control_path).expanduser() if self.semantic_control_path is not None else default_semantic
            if not spatial_path.exists():
                raise FileNotFoundError(f"CtrlRegen spatial control path does not exist: {spatial_path}")
            if not semantic_path.exists():
                raise FileNotFoundError(f"CtrlRegen semantic control path does not exist: {semantic_path}")
            spatial_ref = str(spatial_path.resolve())
            spatial_subfolder = None
            spatial_local_only = True
            semantic_ref = str(semantic_path.resolve())
            semantic_subfolder = "models"
            semantic_local_only = True

        return {
            "source_root": source_root,
            "base_ref": base_ref,
            "base_repo": base_repo,
            "base_downloaded": base_downloaded,
            "encoder_ref": encoder_ref,
            "encoder_repo": encoder_repo,
            "encoder_downloaded": encoder_downloaded,
            "vae_ref": vae_ref,
            "vae_repo": vae_repo,
            "vae_downloaded": vae_downloaded,
            "spatial_ref": spatial_ref,
            "spatial_subfolder": spatial_subfolder,
            "spatial_local_only": spatial_local_only,
            "semantic_ref": semantic_ref,
            "semantic_subfolder": semantic_subfolder,
            "semantic_local_only": semantic_local_only,
            "adapter_repo": DEFAULT_CTRLREGEN_ADAPTER_REPO_ID,
            "adapter_downloaded": adapter_downloaded,
        }

    def _ensure_pipe(self, device: str) -> None:
        if self._pipe is not None and self._pipe_device == device:
            return
        _ensure_torch_diffusers_env()
        import numpy as np
        import torch

        if torch.device(device).type == "mps":
            device = "cpu"
        assets = self._resolve_ctrlregen_assets()
        source_root: Path = assets["source_root"]
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        backup_custom_ip = sys.modules.get("custom_ip_adapter")
        try:
            custom_ip_module = _load_module_from_file(
                "ctrlregen_custom_ip_adapter",
                source_root / "custom_ip_adapter.py",
            )
            sys.modules["custom_ip_adapter"] = custom_ip_module
            custom_i2i_module = _load_module_from_file(
                "ctrlregen_custom_i2i",
                source_root / "custom_i2i_pipeline.py",
            )
        finally:
            if backup_custom_ip is not None:
                sys.modules["custom_ip_adapter"] = backup_custom_ip
            else:
                sys.modules.pop("custom_ip_adapter", None)

        try:
            from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:
            raise RuntimeError("CtrlRegen requires diffusers and transformers in the active environment") from exc

        torch_dtype = _torch_dtype_from_name(self.dtype, device)
        CustomPipe = custom_i2i_module.CustomStableDiffusionControlNetImg2ImgPipeline
        spatialnet = [
            ControlNetModel.from_pretrained(
                assets["spatial_ref"],
                subfolder=assets["spatial_subfolder"],
                torch_dtype=torch_dtype,
                local_files_only=assets["spatial_local_only"],
            )
        ]
        pipe = CustomPipe.from_pretrained(
            assets["base_ref"],
            controlnet=spatialnet,
            torch_dtype=torch_dtype,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.image_encoder = AutoModel.from_pretrained(
            assets["encoder_ref"],
            local_files_only=True,
        ).to(device, dtype=torch_dtype)
        pipe.feature_extractor = AutoImageProcessor.from_pretrained(
            assets["encoder_ref"],
            local_files_only=True,
        )
        try:
            pipe.costum_load_ip_adapter(
                assets["semantic_ref"],
                subfolder=assets["semantic_subfolder"],
                weight_name=self.semantic_control_name,
                image_encoder_folder=None,
                local_files_only=assets["semantic_local_only"],
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "CtrlRegen adapter/base mismatch. Use the official CtrlRegen SD1.5-family base model."
            ) from exc
        pipe.vae = AutoencoderKL.from_pretrained(
            assets["vae_ref"],
            torch_dtype=torch_dtype,
            local_files_only=True,
        ).to(device, dtype=torch_dtype)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_ip_adapter_scale(1.0)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)

        try:
            from controlnet_aux import CannyDetector

            canny = CannyDetector()

            def _run_canny(inp: Image.Image) -> Image.Image:
                out = canny(inp, low_threshold=self.low_threshold, high_threshold=self.high_threshold)
                if isinstance(out, Image.Image):
                    return out.convert("RGB")
                return Image.fromarray(np.asarray(out)).convert("RGB")

            self._canny_impl = _run_canny
        except Exception:
            import cv2

            def _run_canny(inp: Image.Image) -> Image.Image:
                arr = np.asarray(inp.convert("RGB"))
                edges = cv2.Canny(arr, self.low_threshold, self.high_threshold)
                return Image.fromarray(edges).convert("RGB")

            self._canny_impl = _run_canny

        try:
            from color_matcher import ColorMatcher
            from color_matcher.normalizer import Normalizer

            cm = ColorMatcher()

            def _match(ref_img: Image.Image, src_img: Image.Image) -> Image.Image:
                ref_np = Normalizer(np.asarray(ref_img)).type_norm()
                src_np = Normalizer(np.asarray(src_img)).type_norm()
                out_np = cm.transfer(src=src_np, ref=ref_np, method="hm-mkl-hm")
                out_np = Normalizer(out_np).uint8_norm()
                return Image.fromarray(out_np)

            self._color_match = _match
        except Exception:
            self._color_match = lambda _ref_img, src_img: src_img

        self._pipe = pipe
        self._pipe_device = device
        self._resolved_paths = assets
        self._resolved_paths["dtype"] = str(torch_dtype).replace("torch.", "")

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        import torch

        device = context.device or "cpu"
        self._ensure_pipe(device)
        assert self._pipe is not None
        assert self._canny_impl is not None
        assert self._color_match is not None

        image = Image.open(input_path)
        input_size = image.size
        watermarked = _square_image(image, self.image_size, self.square_mode)
        control_img = self._canny_impl(watermarked)
        generator = torch.Generator(device=_generator_device(torch.device(self._pipe_device or device))).manual_seed(
            self._seed_for_context(context)
        )

        with torch.inference_mode():
            attacked = self._pipe(
                self.prompt,
                negative_prompt=self.negative_prompt,
                image=[watermarked],
                control_image=[control_img],
                ip_adapter_image=[watermarked],
                strength=self.step,
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                guidance_scale=self.guidance_scale,
                control_guidance_start=0.0,
                control_guidance_end=1.0,
            ).images[0]

        try:
            attacked = self._color_match(watermarked, attacked)
        except Exception:
            pass
        attacked = _square_image(attacked.convert("RGB"), self.image_size, "fit")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        attacked.save(output_path)
        if str(self._pipe_device or device).startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "backend": "ctrlregen",
            "source_root": str(self._resolved_paths.get("source_root")),
            "base_model": self._resolved_paths.get("base_ref"),
            "base_repo_id": self._resolved_paths.get("base_repo"),
            "base_downloaded": self._resolved_paths.get("base_downloaded"),
            "adapter_repo_id": self._resolved_paths.get("adapter_repo"),
            "adapter_downloaded": self._resolved_paths.get("adapter_downloaded"),
            "image_encoder": self._resolved_paths.get("encoder_ref"),
            "image_encoder_repo_id": self._resolved_paths.get("encoder_repo"),
            "image_encoder_downloaded": self._resolved_paths.get("encoder_downloaded"),
            "vae_model": self._resolved_paths.get("vae_ref"),
            "vae_repo_id": self._resolved_paths.get("vae_repo"),
            "vae_downloaded": self._resolved_paths.get("vae_downloaded"),
            "dtype": self._resolved_paths.get("dtype"),
            "step": self.step,
            "seed": self._seed_for_context(context),
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            "image_size": self.image_size,
            "input_size": list(input_size),
            "output_size": list(attacked.size),
        }


@register_attack
class ImageToVedioAttack(BaseAttack):
    name = "image_to_vedio"
    description = "NFPA image-to-video next-frame prediction attack."

    def __init__(
        self,
        source_root: str | Path | None = None,
        model_root: str | Path | None = None,
        model_id_or_path: str | Path | None = None,
        weight_root: str | Path | None = None,
        image_size: int = 512,
        square_mode: str = "fit",
        num_inference_steps: int = 10,
        xy: int = 40,
        seed: int | None = 1234,
        enforce_model_image_size: bool = True,
        allow_download: bool = True,
        local_files_only: bool = True,
        dtype: str = "auto",
    ) -> None:
        if model_id_or_path is not None and model_root is None:
            model_root = model_id_or_path
        image_size = int(image_size)
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        super().__init__(
            source_root=str(source_root) if source_root is not None else str(DEFAULT_IMAGE_TO_VEDIO_SOURCE_ROOT),
            model_root=str(model_root) if model_root is not None else str(DEFAULT_DIFFUSION_MODEL_ROOT),
            weight_root=str(weight_root) if weight_root is not None else str(DEFAULT_WEIGHT_ROOT),
            image_size=image_size,
            square_mode=square_mode,
            num_inference_steps=num_inference_steps,
            xy=int(xy),
            seed=seed,
            enforce_model_image_size=bool(enforce_model_image_size),
            allow_download=bool(allow_download),
            local_files_only=bool(local_files_only),
            dtype=dtype,
        )
        self.source_root = source_root
        self.model_root = model_root
        self.weight_root = weight_root
        self.image_size = image_size
        self.square_mode = square_mode
        self.num_inference_steps = num_inference_steps
        self.xy = int(xy)
        self.seed = seed
        self.enforce_model_image_size = bool(enforce_model_image_size)
        self.allow_download = bool(allow_download)
        self.local_files_only = bool(local_files_only)
        self.dtype = dtype
        self._pipe: Any | None = None
        self._pipe_device: str | None = None
        self._ddim_scheduler_cls: Any | None = None
        self._ddim_inverse_cls: Any | None = None
        self._model_path: Path | None = None
        self._model_downloaded = False
        self._model_repo_id: str | None = None
        self._model_image_size: int | None = None
        self._torch_dtype: Any | None = None
        self._source_root: Path | None = None

    def _seed_for_context(self, context: AttackContext) -> int:
        if self.seed is not None:
            return int(self.seed)
        if context.seed is not None:
            return int(context.seed)
        return 1234

    def _ensure_pipe(self, device: str) -> None:
        if self._pipe is not None and self._pipe_device == device:
            return
        _ensure_torch_diffusers_env()
        import torch

        source_root = _ensure_backend_source_root(
            self.source_root,
            DEFAULT_IMAGE_TO_VEDIO_SOURCE_ROOT,
            ("utils.py",),
            "NFPA",
        )
        nfpa_utils = _load_module_from_file("nfpa_utils_module", source_root / "utils.py")
        try:
            from diffusers import DDIMInverseScheduler, DDIMScheduler
        except Exception as exc:
            raise RuntimeError("NFPA requires diffusers in the active environment") from exc

        model_path, downloaded, repo_id = _resolve_diffusion_model_root(
            self.model_root,
            self.weight_root,
            self.allow_download,
        )
        torch_dtype = _torch_dtype_from_name(self.dtype, device)
        MyStableDiffusionPipeline = nfpa_utils.MyStableDiffusionPipeline
        pipe = MyStableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            local_files_only=self.local_files_only and not downloaded,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.unet.requires_grad_(False)

        sample_size = int(getattr(pipe.unet.config, "sample_size", 0) or 0)
        vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
        model_image_size = sample_size * vae_scale_factor if sample_size > 0 else None
        if model_image_size is not None and self.image_size != model_image_size:
            message = (
                f"NFPA model/image-size mismatch: model expects {model_image_size}x{model_image_size} "
                f"(UNet sample_size={sample_size}), but image_size={self.image_size}."
            )
            if self.enforce_model_image_size:
                raise ValueError(message)

        self._pipe = pipe
        self._pipe_device = device
        self._ddim_scheduler_cls = DDIMScheduler
        self._ddim_inverse_cls = DDIMInverseScheduler
        self._model_path = model_path
        self._model_downloaded = downloaded
        self._model_repo_id = repo_id
        self._model_image_size = model_image_size
        self._torch_dtype = torch_dtype
        self._source_root = source_root

    def _invert_latents(self, image_tensor: Any, seed: int) -> Any:
        import torch

        assert self._pipe is not None
        assert self._ddim_inverse_cls is not None
        assert self._ddim_scheduler_cls is not None
        image_tensor = image_tensor.to(self._pipe.device, dtype=self._pipe.unet.dtype)
        image_tensor = image_tensor * 2.0 - 1.0
        with torch.no_grad():
            latents = self._pipe.vae.encode(image_tensor)["latent_dist"].mean
            latents = latents * self._pipe.vae.config.scaling_factor
            self._pipe.scheduler = self._ddim_inverse_cls.from_config(self._pipe.scheduler.config)
            generator = torch.Generator(device=_generator_device(self._pipe.device)).manual_seed(seed)
            inversed = self._pipe(
                prompt="",
                negative_prompt="",
                num_inference_steps=self.num_inference_steps,
                latents=latents,
                output_type="latent",
                width=self.image_size,
                height=self.image_size,
                generator=generator,
            ).images
            self._pipe.scheduler = self._ddim_scheduler_cls.from_config(self._pipe.scheduler.config)
        return inversed

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        import torch
        from torchvision import transforms

        device = context.device or "cpu"
        self._ensure_pipe(device)
        assert self._pipe is not None
        assert self._model_path is not None

        image = Image.open(input_path)
        input_size = image.size
        watermarked = _square_image(image, self.image_size, self.square_mode)
        tensor = transforms.ToTensor()(watermarked).unsqueeze(0)
        seed = self._seed_for_context(context)
        inversed_latents = self._invert_latents(tensor, seed)
        warped_timestep = torch.tensor([0], dtype=torch.long, device=self._pipe.device)
        generator = torch.Generator(device=_generator_device(self._pipe.device)).manual_seed(seed)
        with torch.no_grad():
            images = self._pipe(
                prompt="",
                num_images_per_prompt=2,
                latents=inversed_latents,
                xyz=[self.xy, self.xy, 0],
                num_inference_steps=self.num_inference_steps,
                warped_latents_timestep=warped_timestep,
                generator=generator,
            ).images
        attacked = _square_image(images[1].convert("RGB"), self.image_size, "fit")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        attacked.save(output_path)
        if str(self._pipe_device or device).startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "backend": "nfpa",
            "source_root": str(self._source_root),
            "model_repo_id": self._model_repo_id,
            "model_path": str(self._model_path),
            "model_downloaded": self._model_downloaded,
            "model_image_size": self._model_image_size,
            "dtype": str(self._torch_dtype).replace("torch.", ""),
            "xy": self.xy,
            "seed": seed,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size,
            "input_size": list(input_size),
            "output_size": list(attacked.size),
        }
