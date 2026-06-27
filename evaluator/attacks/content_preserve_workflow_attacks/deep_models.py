from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image


ModelKind = Literal["rrdbnet_x4", "swinir_jpeg_car", "restormer_denoise"]
_MODEL_CACHE: dict[tuple[ModelKind, str, str], nn.Module] = {}


def find_checkpoint(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Weight path does not exist: {path}")
    candidates = sorted(path.glob("*.pth")) + sorted(path.glob("*.pt")) + sorted(path.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under: {path}")
    return candidates[0]


def select_device(requested: str | None) -> torch.device:
    requested = (requested or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("mps") and torch.backends.mps.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def tensor_to_image(tensor: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    import numpy as np

    tensor = tensor.detach().float().clamp_(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu()
    array = (tensor.numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (height, width)


def crop_to_size(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    return tensor[:, :, :height, :width]


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    return checkpoint


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    b = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(b, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, to_2tuple(window_size), num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.register_buffer("attn_mask", self.calculate_mask(input_resolution) if shift_size > 0 else None)

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, height, width, c)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, c)
        if self.input_resolution == x_size:
            mask = self.attn_mask
        else:
            mask = self.calculate_mask(x_size).to(x.device)
        attn_windows = self.attn(x_windows, mask=mask)
        shifted_x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, c), self.window_size, height, width)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = shortcut + x.view(b, height * width, c)
        return x + self.mlp(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, x_size)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 1,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim: int = 96) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        img_size: int | tuple[int, int],
    ) -> None:
        super().__init__()
        self.residual_group = BasicLayer(dim, input_resolution, depth, num_heads, window_size, mlp_ratio)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=1, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        residual = self.residual_group(x, x_size)
        return self.patch_embed(self.conv(self.patch_unembed(residual, x_size))) + x


class SwinIRJPEG(nn.Module):
    def __init__(
        self,
        img_size: int = 126,
        window_size: int = 7,
        img_range: float = 1.0,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        embed_dim: int = 180,
        num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.img_range = img_range
        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        self.conv_first = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=1, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
        resolution = tuple(self.patch_embed.patches_resolution)
        self.pos_drop = nn.Dropout(0.0)
        self.layers = nn.ModuleList(
            [
                RSTB(embed_dim, resolution, depths[index], num_heads[index], window_size, mlp_ratio, img_size)
                for index in range(len(depths))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        x = x + self.conv_last(res)
        return x / self.img_range + mean


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = BiasFreeLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return rearrange(self.body(rearrange(x, "b c h w -> b (h w) c")), "b (h w) c -> b c h w", h=height, w=width)


class RestormerFeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class RestormerAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = attn.softmax(dim=-1) @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class RestormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion: float = 2.66) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = RestormerAttention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = RestormerFeedForward(dim, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 48) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def make_restormer_stage(dim: int, count: int, heads: int) -> nn.Sequential:
    return nn.Sequential(*[RestormerBlock(dim, heads) for _ in range(count)])


class RestormerDenoise(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(3, 48)
        self.encoder_level1 = make_restormer_stage(48, 4, 1)
        self.down1_2 = Downsample(48)
        self.encoder_level2 = make_restormer_stage(96, 6, 2)
        self.down2_3 = Downsample(96)
        self.encoder_level3 = make_restormer_stage(192, 6, 4)
        self.down3_4 = Downsample(192)
        self.latent = make_restormer_stage(384, 8, 8)
        self.up4_3 = Upsample(384)
        self.reduce_chan_level3 = nn.Conv2d(384, 192, 1, bias=False)
        self.decoder_level3 = make_restormer_stage(192, 6, 4)
        self.up3_2 = Upsample(192)
        self.reduce_chan_level2 = nn.Conv2d(192, 96, 1, bias=False)
        self.decoder_level2 = make_restormer_stage(96, 6, 2)
        self.up2_1 = Upsample(96)
        self.decoder_level1 = make_restormer_stage(96, 4, 1)
        self.refinement = make_restormer_stage(96, 4, 1)
        self.output = nn.Conv2d(96, 3, 3, 1, 1, bias=False)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))
        inp_dec_level3 = torch.cat([self.up4_3(latent), out_enc_level3], dim=1)
        out_dec_level3 = self.decoder_level3(self.reduce_chan_level3(inp_dec_level3))
        inp_dec_level2 = torch.cat([self.up3_2(out_dec_level3), out_enc_level2], dim=1)
        out_dec_level2 = self.decoder_level2(self.reduce_chan_level2(inp_dec_level2))
        inp_dec_level1 = torch.cat([self.up2_1(out_dec_level2), out_enc_level1], dim=1)
        out_dec_level1 = self.refinement(self.decoder_level1(inp_dec_level1))
        return self.output(out_dec_level1) + inp_img


def _load_model(kind: ModelKind, weight_path: Path, device: torch.device) -> nn.Module:
    checkpoint = find_checkpoint(weight_path)
    key = (kind, str(checkpoint.resolve()), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if kind == "rrdbnet_x4":
        model: nn.Module = RRDBNet(scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    elif kind == "swinir_jpeg_car":
        model = SwinIRJPEG()
    elif kind == "restormer_denoise":
        model = RestormerDenoise()
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    state = load_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def run_rrdbnet_x4(image: Image.Image, weight_path: Path, device_name: str | None = None) -> Image.Image:
    device = select_device(device_name)
    model = _load_model("rrdbnet_x4", weight_path, device)
    tensor = image_to_tensor(image, device)
    with torch.inference_mode():
        output = model(tensor)
    return tensor_to_image(output)


def run_swinir_jpeg_car(image: Image.Image, weight_path: Path, device_name: str | None = None) -> Image.Image:
    device = select_device(device_name)
    model = _load_model("swinir_jpeg_car", weight_path, device)
    tensor = image_to_tensor(image, device)
    padded, original_hw = pad_to_multiple(tensor, 7)
    with torch.inference_mode():
        output = crop_to_size(model(padded), original_hw)
    return tensor_to_image(output, size=image.size)


def run_restormer_denoise(image: Image.Image, weight_path: Path, device_name: str | None = None) -> Image.Image:
    device = select_device(device_name)
    model = _load_model("restormer_denoise", weight_path, device)
    tensor = image_to_tensor(image, device)
    padded, original_hw = pad_to_multiple(tensor, 8)
    with torch.inference_mode():
        output = crop_to_size(model(padded), original_hw)
    return tensor_to_image(output, size=image.size)
