from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.nn.init import _calculate_fan_in_and_fan_out


ModelKind = Literal["deepwb_awb", "3dlut_fivek", "retinexformer_lol"]
_MODEL_CACHE: dict[tuple[ModelKind, str, str, tuple[Any, ...]], nn.Module | tuple[nn.Module, torch.Tensor]] = {}


def _select_device(requested: str | None) -> torch.device:
    requested = (requested or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("mps") and torch.backends.mps.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_named_checkpoint(path: Path, filename: str) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        if path.name != filename:
            raise FileNotFoundError(f"Expected {filename}, got checkpoint file: {path}")
        return path
    target = path / filename
    if not target.exists():
        raise FileNotFoundError(f"Expected checkpoint file does not exist: {target}")
    return target


def _find_checkpoint(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Weight path does not exist: {path}")
    candidates = (
        sorted(path.glob("*.pth"))
        + sorted(path.glob("*.pt"))
        + sorted(path.glob("*.ckpt"))
        + sorted(path.glob("*.bin"))
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under: {path}")
    return candidates[0]


def _strip_state_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped = key
        for prefix in ("module.", "net_g.", "network_g."):
            if mapped.startswith(prefix):
                mapped = mapped[len(prefix) :]
        normalized[mapped] = value
    return normalized


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("params_ema", "params", "state_dict", "model", "network_g", "net"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _strip_state_prefix(value)
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return _strip_state_prefix(checkpoint)
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")


def _image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def _image_to_tensor_cpu(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _move_batch_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    non_blocking = False
    if device.type == "cuda":
        try:
            tensor = tensor.pin_memory()
            non_blocking = True
        except Exception:
            non_blocking = False
    return tensor.to(device=device, dtype=torch.float32, non_blocking=non_blocking)


def _tensor_to_image(tensor: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    tensor = tensor.detach().float().clamp_(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu()
    array = (tensor.numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"), (height, width)


def _crop_to_hw(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    return tensor[:, :, :height, :width]


def _group_indices_by_shape(tensors: list[torch.Tensor]) -> dict[tuple[int, ...], list[int]]:
    grouped: dict[tuple[int, ...], list[int]] = {}
    for index, tensor in enumerate(tensors):
        grouped.setdefault(tuple(tensor.shape), []).append(index)
    return grouped


class DeepWBDoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DeepWBDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DeepWBDoubleConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class DeepWBBridgeDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class DeepWBBridgeUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_up(x)


class DeepWBUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DeepWBDoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x2, x1], dim=1)
        return torch.relu(self.up(self.conv(x)))


class DeepWBOutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_conv = nn.Sequential(
            DeepWBDoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.out_conv(torch.cat([x2, x1], dim=1))


class DeepWBAWBNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_channels = 3
        self.encoder_inc = DeepWBDoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DeepWBDownBlock(24, 48)
        self.encoder_down2 = DeepWBDownBlock(48, 96)
        self.encoder_down3 = DeepWBDownBlock(96, 192)
        self.encoder_bridge_down = DeepWBBridgeDown(192, 384)
        self.decoder_bridge_up = DeepWBBridgeUp(384, 192)
        self.decoder_up1 = DeepWBUpBlock(192, 96)
        self.decoder_up2 = DeepWBUpBlock(96, 48)
        self.decoder_up3 = DeepWBUpBlock(48, 24)
        self.decoder_out = DeepWBOutputBlock(24, self.n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x = self.decoder_up3(x, x2)
        return self.decoder_out(x, x1)


def _deepwb_kernel(image: np.ndarray) -> np.ndarray:
    return np.transpose(
        (
            image[:, 0],
            image[:, 1],
            image[:, 2],
            image[:, 0] * image[:, 1],
            image[:, 0] * image[:, 2],
            image[:, 1] * image[:, 2],
            image[:, 0] * image[:, 0],
            image[:, 1] * image[:, 1],
            image[:, 2] * image[:, 2],
            image[:, 0] * image[:, 1] * image[:, 2],
            np.ones(image.shape[0], dtype=image.dtype),
        )
    )


def _deepwb_get_mapping_func(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_flat = source.reshape(-1, 3).astype(np.float64)
    target_flat = target.reshape(-1, 3).astype(np.float64)
    mapping, *_ = np.linalg.lstsq(_deepwb_kernel(source_flat), target_flat, rcond=None)
    return mapping


def _deepwb_apply_mapping_func(image: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    shape = image.shape
    image_flat = image.reshape(-1, 3).astype(np.float64)
    result = _deepwb_kernel(image_flat) @ mapping
    return result.reshape(shape)


def _load_deepwb_awb(weight_path: Path, device: torch.device) -> nn.Module:
    checkpoint = _find_named_checkpoint(weight_path, "net_awb.pth")
    key = ("deepwb_awb", str(checkpoint.resolve()), str(device), ())
    cached = _MODEL_CACHE.get(key)
    if isinstance(cached, nn.Module):
        return cached
    model = DeepWBAWBNet()
    model.load_state_dict(_extract_state_dict(_torch_load(checkpoint)), strict=True)
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def run_deepwb_awb(
    image: Image.Image,
    weight_path: Path,
    device_name: str | None = None,
    max_size: int = 656,
) -> Image.Image:
    device = _select_device(device_name)
    model = _load_deepwb_awb(weight_path, device)
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    scale = float(max_size) / float(max(image.size))
    resized_size = (max(16, round(image.width * scale)), max(16, round(image.height * scale)))
    resized_size = (
        resized_size[0] if resized_size[0] % 16 == 0 else resized_size[0] + 16 - resized_size[0] % 16,
        resized_size[1] if resized_size[1] % 16 == 0 else resized_size[1] + 16 - resized_size[1] % 16,
    )
    resized = image.convert("RGB").resize(resized_size, Image.Resampling.BICUBIC)
    resized_array = np.asarray(resized, dtype=np.float32)
    tensor = _image_to_tensor(resized, device)
    with torch.inference_mode():
        output = model(tensor)
    output_array = output.detach().float().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    mapping = _deepwb_get_mapping_func(resized_array, output_array)
    mapped = np.clip(_deepwb_apply_mapping_func(original, mapping), 0.0, 1.0)
    return Image.fromarray((mapped * 255.0).round().astype(np.uint8)).convert("RGB")


def _deepwb_resized_size(image: Image.Image, max_size: int) -> tuple[int, int]:
    scale = float(max_size) / float(max(image.size))
    resized_size = (max(16, round(image.width * scale)), max(16, round(image.height * scale)))
    return (
        resized_size[0] if resized_size[0] % 16 == 0 else resized_size[0] + 16 - resized_size[0] % 16,
        resized_size[1] if resized_size[1] % 16 == 0 else resized_size[1] + 16 - resized_size[1] % 16,
    )


def run_deepwb_awb_batch(
    images: list[Image.Image],
    weight_path: Path,
    device_name: str | None = None,
    max_size: int = 656,
) -> list[Image.Image]:
    if not images:
        return []
    device = _select_device(device_name)
    model = _load_deepwb_awb(weight_path, device)

    originals: list[np.ndarray] = []
    resized_arrays: list[np.ndarray] = []
    tensors: list[torch.Tensor] = []
    for image in images:
        original = np.asarray(image.convert("RGB"), dtype=np.float32)
        resized = image.convert("RGB").resize(_deepwb_resized_size(image, max_size), Image.Resampling.BICUBIC)
        originals.append(original)
        resized_arrays.append(np.asarray(resized, dtype=np.float32))
        tensors.append(_image_to_tensor_cpu(resized))

    results: list[Image.Image | None] = [None] * len(images)
    with torch.inference_mode():
        for indices in _group_indices_by_shape(tensors).values():
            batch = _move_batch_to_device(torch.stack([tensors[index] for index in indices], dim=0), device)
            output = model(batch).detach().float().clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
            for batch_index, image_index in enumerate(indices):
                mapping = _deepwb_get_mapping_func(resized_arrays[image_index], output[batch_index])
                mapped = np.clip(_deepwb_apply_mapping_func(originals[image_index], mapping), 0.0, 1.0)
                results[image_index] = Image.fromarray((mapped * 255.0).round().astype(np.uint8)).convert("RGB")
    return [result for result in results if result is not None]


def _lut_discriminator_block(in_filters: int, out_filters: int, normalization: bool = False) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class LUTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear"),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *_lut_discriminator_block(16, 32, normalization=True),
            *_lut_discriminator_block(32, 64, normalization=True),
            *_lut_discriminator_block(64, 128, normalization=True),
            *_lut_discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def _load_3dlut(weight_path: Path, device: torch.device) -> tuple[nn.Module, torch.Tensor]:
    classifier_path = _find_named_checkpoint(weight_path, "classifier.pth")
    luts_path = _find_named_checkpoint(weight_path, "LUTs.pth")
    key = ("3dlut_fivek", str(weight_path.resolve()), str(device), ())
    cached = _MODEL_CACHE.get(key)
    if isinstance(cached, tuple):
        return cached

    classifier = LUTClassifier()
    classifier.load_state_dict(_extract_state_dict(_torch_load(classifier_path)), strict=True)
    classifier.to(device).eval()

    raw_luts = _torch_load(luts_path)
    if not isinstance(raw_luts, Mapping):
        raise TypeError(f"Unsupported 3DLUT checkpoint type: {type(raw_luts).__name__}")
    lut_tensors = []
    for index in ("0", "1", "2"):
        entry = raw_luts.get(index)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("LUT"), torch.Tensor):
            raise KeyError(f"3DLUT checkpoint is missing LUT {index}")
        lut_tensors.append(entry["LUT"].to(device=device, dtype=torch.float32))
    luts = torch.stack(lut_tensors, dim=0)
    result: tuple[nn.Module, torch.Tensor] = (classifier, luts)
    _MODEL_CACHE[key] = result
    return result


def _trilinear_lut(lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    channels, dim_r, dim_g, dim_b = lut.shape
    if channels != 3 or dim_r != dim_g or dim_r != dim_b:
        raise ValueError(f"Unsupported LUT shape: {tuple(lut.shape)}")
    image = image.clamp(0.0, 1.0)
    coords = image.permute(0, 2, 3, 1)
    grid = torch.stack(
        (
            coords[..., 2] * 2.0 - 1.0,
            coords[..., 1] * 2.0 - 1.0,
            coords[..., 0] * 2.0 - 1.0,
        ),
        dim=-1,
    ).unsqueeze(1)
    volume = lut.unsqueeze(0).expand(image.shape[0], -1, -1, -1, -1)
    output = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return output.squeeze(2)


def _trilinear_lut_batch(luts: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    if luts.ndim != 5:
        raise ValueError(f"Unsupported batched LUT shape: {tuple(luts.shape)}")
    batch, channels, dim_r, dim_g, dim_b = luts.shape
    if channels != 3 or dim_r != dim_g or dim_r != dim_b:
        raise ValueError(f"Unsupported batched LUT shape: {tuple(luts.shape)}")
    if images.shape[0] != batch:
        raise ValueError(f"LUT batch size {batch} does not match image batch size {images.shape[0]}")

    images = images.clamp(0.0, 1.0)
    coords = images.permute(0, 2, 3, 1)
    grid = torch.stack(
        (
            coords[..., 2] * 2.0 - 1.0,
            coords[..., 1] * 2.0 - 1.0,
            coords[..., 0] * 2.0 - 1.0,
        ),
        dim=-1,
    ).unsqueeze(1)
    output = F.grid_sample(luts, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return output.squeeze(2)


def run_image_adaptive_3dlut(
    image: Image.Image,
    weight_path: Path,
    device_name: str | None = None,
    blend: float = 1.0,
) -> Image.Image:
    device = _select_device(device_name)
    classifier, luts = _load_3dlut(weight_path, device)
    tensor = _image_to_tensor(image, device)
    with torch.inference_mode():
        weights = classifier(tensor).view(tensor.shape[0], 3)
        combined = torch.einsum("nk,kcrgb->ncrgb", weights, luts)
        outputs = []
        for idx in range(tensor.shape[0]):
            outputs.append(_trilinear_lut(combined[idx], tensor[idx : idx + 1]))
        output = torch.cat(outputs, dim=0)
        if blend < 1.0:
            output = tensor * (1.0 - blend) + output * blend
    return _tensor_to_image(output, size=image.size)


def run_image_adaptive_3dlut_batch(
    images: list[Image.Image],
    weight_path: Path,
    device_name: str | None = None,
    blend: float = 1.0,
) -> list[Image.Image]:
    if not images:
        return []
    device = _select_device(device_name)
    classifier, luts = _load_3dlut(weight_path, device)
    tensors = [_image_to_tensor_cpu(image) for image in images]
    results: list[Image.Image | None] = [None] * len(images)

    with torch.inference_mode():
        for indices in _group_indices_by_shape(tensors).values():
            batch = _move_batch_to_device(torch.stack([tensors[index] for index in indices], dim=0), device)
            weights = classifier(batch).view(batch.shape[0], 3)
            combined = torch.einsum("nk,kcrgb->ncrgb", weights, luts)
            output = _trilinear_lut_batch(combined, batch)
            if blend < 1.0:
                output = batch * (1.0 - blend) + output * blend
            output = output.detach().cpu()
            for batch_index, image_index in enumerate(indices):
                results[image_index] = _tensor_to_image(output[batch_index : batch_index + 1], size=images[image_index].size)
    return [result for result in results if result is not None]


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float, std: float, a: float, b: float) -> torch.Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class RetinexPreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class RetinexGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class IlluminationEstimator(nn.Module):
    def __init__(self, n_fea_middle: int, n_fea_in: int = 4, n_fea_out: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_c = image.mean(dim=1, keepdim=True)
        input_tensor = torch.cat([image, mean_c], dim=1)
        x_1 = self.conv1(input_tensor)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IGMSA(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            RetinexGELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x_in.shape
        x = x_in.reshape(batch, height * width, channels)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v, illu_attn = map(
            lambda tensor: rearrange(tensor, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_fea_trans.flatten(1, 2)),
        )
        v = v * illu_attn
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).permute(0, 3, 1, 2)
        x = x.reshape(batch, height * width, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(batch, height, width, channels)
        out_p = self.pos_emb(v_inp.reshape(batch, height, width, channels).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class RetinexFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            RetinexGELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            RetinexGELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.ModuleList([IGMSA(dim=dim, dim_head=dim_head, heads=heads), RetinexPreNorm(dim, RetinexFeedForward(dim=dim))]) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        illu_fea_trans = illu_fea.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea_trans) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class RetinexDenoiser(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 3, dim: int = 40, level: int = 2, num_blocks: list[int] | None = None) -> None:
        super().__init__()
        num_blocks = num_blocks or [1, 2, 2]
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList()
        dim_level = dim
        for index in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(dim=dim_level, num_blocks=num_blocks[index], dim_head=dim, heads=dim_level // dim),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_level *= 2

        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList()
        for index in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - index],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                    ]
                )
            )
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        fea = self.embedding(x)
        fea_encoder = []
        illu_fea_list = []
        for igab, fea_downsample, illu_fea_downsample in self.encoder_layers:
            fea = igab(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = fea_downsample(fea)
            illu_fea = illu_fea_downsample(illu_fea)

        fea = self.bottleneck(fea, illu_fea)
        for index, (fea_upsample, fusion, igab) in enumerate(self.decoder_layers):
            fea = fea_upsample(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.level - 1 - index]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - index]
            fea = igab(fea, illu_fea)
        return self.mapping(fea) + x


class RetinexFormerSingleStage(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feat: int = 40, level: int = 2, num_blocks: list[int] | None = None) -> None:
        super().__init__()
        self.estimator = IlluminationEstimator(n_feat)
        self.denoiser = RetinexDenoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks or [1, 2, 2])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(image)
        input_image = image * illu_map + image
        return self.denoiser(input_image, illu_fea)


class RetinexFormer(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feat: int = 40, stage: int = 1, num_blocks: list[int] | None = None) -> None:
        super().__init__()
        self.stage = stage
        self.body = nn.Sequential(
            *[
                RetinexFormerSingleStage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_feat=n_feat,
                    level=2,
                    num_blocks=num_blocks or [1, 2, 2],
                )
                for _ in range(stage)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _infer_retinex_config(state: Mapping[str, torch.Tensor]) -> tuple[int, int, list[int]]:
    conv1 = state.get("body.0.estimator.conv1.weight")
    n_feat = int(conv1.shape[0]) if isinstance(conv1, torch.Tensor) else 40
    stage_indices = []
    for key in state:
        if key.startswith("body."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                stage_indices.append(int(parts[1]))
    stage = max(stage_indices) + 1 if stage_indices else 1
    return n_feat, stage, [1, 2, 2]


def _load_retinexformer(weight_path: Path, device: torch.device) -> nn.Module:
    checkpoint = _find_checkpoint(weight_path)
    state = _extract_state_dict(_torch_load(checkpoint))
    n_feat, stage, num_blocks = _infer_retinex_config(state)
    key = ("retinexformer_lol", str(checkpoint.resolve()), str(device), (n_feat, stage, tuple(num_blocks)))
    cached = _MODEL_CACHE.get(key)
    if isinstance(cached, nn.Module):
        return cached
    model = RetinexFormer(n_feat=n_feat, stage=stage, num_blocks=num_blocks)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def run_retinexformer_low_light(
    image: Image.Image,
    weight_path: Path,
    device_name: str | None = None,
    window_size: int = 4,
) -> Image.Image:
    device = _select_device(device_name)
    model = _load_retinexformer(weight_path, device)
    tensor = _image_to_tensor(image, device)
    padded, original_hw = _pad_to_multiple(tensor, window_size)
    with torch.inference_mode():
        output = _crop_to_hw(model(padded), original_hw)
    return _tensor_to_image(output, size=image.size)


def run_retinexformer_low_light_batch(
    images: list[Image.Image],
    weight_path: Path,
    device_name: str | None = None,
    window_size: int = 4,
) -> list[Image.Image]:
    if not images:
        return []
    device = _select_device(device_name)
    model = _load_retinexformer(weight_path, device)
    padded_tensors: list[torch.Tensor] = []
    original_hws: list[tuple[int, int]] = []
    for image in images:
        padded, original_hw = _pad_to_multiple(_image_to_tensor_cpu(image).unsqueeze(0), window_size)
        padded_tensors.append(padded.squeeze(0))
        original_hws.append(original_hw)

    results: list[Image.Image | None] = [None] * len(images)
    with torch.inference_mode():
        for indices in _group_indices_by_shape(padded_tensors).values():
            batch = _move_batch_to_device(torch.stack([padded_tensors[index] for index in indices], dim=0), device)
            output = model(batch).detach().cpu()
            for batch_index, image_index in enumerate(indices):
                cropped = _crop_to_hw(output[batch_index : batch_index + 1], original_hws[image_index])
                results[image_index] = _tensor_to_image(cropped, size=images[image_index].size)
    return [result for result in results if result is not None]
