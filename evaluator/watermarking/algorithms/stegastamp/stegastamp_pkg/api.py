import os
import sys
from typing import Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

try:
    import bchlib  # type: ignore
except Exception:  # pragma: no cover
    bchlib = None

from .model_def import StegaStampEncoder, StegaStampDecoder


IMAGE_SIZE = 400
SECRET_SIZE = 100


def _ensure_device(device: Optional[str] = None) -> torch.device:
    # 允许外部传入 'cuda'/'cpu'，但若请求 cuda 而不可用，则回退到 cpu
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return torch.device(device)


def _to_tensor_image(img: Image.Image) -> torch.Tensor:
    t = transforms.ToTensor()
    return t(img)


def _prepare_image(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        img = img.convert("RGB")
    else:
        raise TypeError("Unsupported image type. Use path, PIL.Image, or numpy array.")
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC)


def _bch_setup():
    """Return a BCH encoder instance or None if unavailable/invalid.
    Gracefully handles environments where bchlib exists but cannot init
    with given parameters (e.g., platform-specific builds).
    """
    if bchlib is None:
        return None
    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    try:
        return bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    except Exception:
        return None


def _secret_from_string(secret: str, device: torch.device) -> torch.Tensor:
    """Encode up to 7 ASCII characters into 100-bit tensor using BCH ECC.
    If longer than 7, raises ValueError.
    """
    bch = _bch_setup()
    if len(secret) > 7:
        raise ValueError("Can only encode up to 7 ASCII characters with ECC (56 bits).")
    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    bits = [int(x) for x in packet_binary]
    # pad to SECRET_SIZE
    if len(bits) > SECRET_SIZE:
        raise ValueError(f"Encoded bits exceed SECRET_SIZE {SECRET_SIZE}.")
    bits = bits + [0] * (SECRET_SIZE - len(bits))
    arr = torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)
    return arr


def _string_from_secret_bits(bits: np.ndarray) -> Optional[str]:
    bch = _bch_setup()
    if bch is None:
        return None
    bits = np.round(bits).astype(np.uint8)
    packet_binary = "".join([str(int(bit)) for bit in bits[:96]])
    packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
    packet = bytearray(packet)
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
    try:
        bitflips = bch.decode_inplace(data, ecc)
    except Exception:
        return None
    if bitflips != -1:
        try:
            return data.decode("utf-8").rstrip()
        except Exception:
            return None
    return None


def _resolve_state_dict(obj):
    import torch.nn as nn
    from collections import OrderedDict

    # torch.save(model) -> nn.Module
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    # torch.save(state_dict)
    if isinstance(obj, (dict, OrderedDict)):
        # common wrappers
        for key in ['state_dict', 'model', 'encoder', 'decoder', 'net', 'weights']:
            if key in obj and isinstance(obj[key], (dict, OrderedDict)):
                return obj[key]
        # raw state_dict
        return obj
    raise ValueError("Unsupported checkpoint format")


def _strip_module_prefix(state_dict):
    if all(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


class StegaStamp:
    """Simple wrapper to load trained StegaStamp encoder/decoder and expose embed/decode APIs."""

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        device: Optional[str] = None,
    ) -> None:
        self.device = _ensure_device(device)

        # Load as state_dict (supports both raw state_dict and full-model pickles)
        def _load_with_shim(path):
            def _do_load() -> object:
                load_kwargs = {"map_location": self.device}
                try:
                    # PyTorch >=2.6 requires weights_only=False to load pickled modules
                    return torch.load(path, weights_only=False, **load_kwargs)
                except TypeError:
                    # Older versions do not accept weights_only argument
                    return torch.load(path, **load_kwargs)

            try:
                return _do_load()
            except Exception:
                # Install a shim module named 'model' exposing training-time class names
                self._install_pickle_shim()
                return _do_load()

        enc_obj = _load_with_shim(encoder_path)
        dec_obj = _load_with_shim(decoder_path)
        enc_sd = _strip_module_prefix(_resolve_state_dict(enc_obj))
        dec_sd = _strip_module_prefix(_resolve_state_dict(dec_obj))

        self.encoder = StegaStampEncoder(secret_size=SECRET_SIZE).to(self.device)
        self.decoder = StegaStampDecoder(secret_size=SECRET_SIZE).to(self.device)
        _ = self.encoder.load_state_dict(enc_sd, strict=False)
        _ = self.decoder.load_state_dict(dec_sd, strict=False)
        # missing/unexpected are NamedTuple in newer torch; no raise here to allow partial load if shapes match
        self.encoder.eval()
        self.decoder.eval()

    def _install_pickle_shim(self) -> None:
        import types
        from .model_def import Dense, Conv2D, Flatten, StegaStampEncoder as Enc, SpatialTransformerNetwork, StegaStampDecoder as Dec
        # create a lightweight 'model' module with the expected class names
        if 'model' in sys.modules:
            existing = sys.modules['model']
            if all(hasattr(existing, attr) for attr in ['StegaStampEncoder', 'StegaStampDecoder']):
                return
        m = types.ModuleType('model')
        m.Dense = Dense
        m.Conv2D = Conv2D
        m.Flatten = Flatten
        m.StegaStampEncoder = Enc
        m.SpatialTransformerNetwork = SpatialTransformerNetwork
        m.StegaStampDecoder = Dec
        # optional: discriminator placeholder for completeness
        class Discriminator(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x
        m.Discriminator = Discriminator
        sys.modules['model'] = m

    @torch.no_grad()
    def embed(
        self,
        image: Union[str, Image.Image, np.ndarray],
        secret: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        return_residual: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Embed a secret into an image. Returns encoded image (and residual if requested).

        secret can be:
          - str: up to 7 ASCII chars, encoded with BCH to 100 bits
          - numpy array / torch tensor: shape (100,) of {0,1}
        """
        device = self.device
        pil_img = _prepare_image(image)
        img_tensor = _to_tensor_image(pil_img).unsqueeze(0).to(device)

        if secret is None:
            bits = np.random.randint(0, 2, size=SECRET_SIZE, dtype=np.uint8)
            secret_tensor = torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)
        elif isinstance(secret, str):
            secret_tensor = _secret_from_string(secret, device)
        elif isinstance(secret, np.ndarray):
            if secret.ndim != 1 or secret.shape[0] != SECRET_SIZE:
                raise ValueError(f"secret bits must be 1-D of length {SECRET_SIZE}")
            secret_tensor = torch.tensor(secret.astype(np.float32), device=device).unsqueeze(0)
        elif isinstance(secret, torch.Tensor):
            if secret.dim() == 1:
                secret_tensor = secret.to(device).float().unsqueeze(0)
            else:
                secret_tensor = secret.to(device).float()
        else:
            raise TypeError("Unsupported secret type.")

        residual = self.encoder((secret_tensor, img_tensor))
        encoded = torch.clamp(img_tensor + residual, 0, 1).squeeze(0).cpu().numpy()
        encoded = (encoded * 255).astype(np.uint8).transpose((1, 2, 0))

        encoded_pil = Image.fromarray(encoded)
        if not return_residual:
            return encoded_pil

        residual_vis = (residual.squeeze(0).cpu().numpy() + 0.5)
        residual_vis = np.clip(residual_vis, 0, 1)
        residual_vis = (residual_vis * 255).astype(np.uint8).transpose((1, 2, 0))
        residual_pil = Image.fromarray(residual_vis)
        return encoded_pil, residual_pil

    @torch.no_grad()
    def decode(
        self,
        image: Union[str, Image.Image, np.ndarray],
        return_bits: bool = False,
    ) -> Union[str, np.ndarray, Tuple[Optional[str], np.ndarray]]:
        """Decode secret from an image. Returns best-effort recovered string (if BCH available)
        and/or the raw 100-bit vector rounded to {0,1}.
        """
        device = self.device
        pil_img = _prepare_image(image)
        img_tensor = _to_tensor_image(pil_img).unsqueeze(0).to(device)

        secret_prob = self.decoder(img_tensor).cpu().numpy()[0]
        bits = np.round(secret_prob).astype(np.uint8)

        if return_bits and bchlib is None:
            return bits

        decoded: Optional[str] = None
        if bchlib is not None:
            decoded = _string_from_secret_bits(bits)

        if return_bits:
            return decoded, bits
        else:
            return decoded if decoded is not None else ""


def load_stegastamp(
    encoder_path: str,
    decoder_path: str,
    device: Optional[str] = None,
) -> StegaStamp:
    return StegaStamp(encoder_path, decoder_path, device=device)


def embed_image(
    encoder_path: str,
    decoder_path: str,
    image: Union[str, Image.Image, np.ndarray],
    secret: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
    device: Optional[str] = None,
    return_residual: bool = False,
) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
    model = load_stegastamp(encoder_path, decoder_path, device=device)
    return model.embed(image, secret, return_residual=return_residual)


def decode_image(
    encoder_path: str,
    decoder_path: str,
    image: Union[str, Image.Image, np.ndarray],
    device: Optional[str] = None,
    return_bits: bool = False,
) -> Union[str, np.ndarray, Tuple[Optional[str], np.ndarray]]:
    model = load_stegastamp(encoder_path, decoder_path, device=device)
    return model.decode(image, return_bits=return_bits)


def embed_file(
    encoder_path: str,
    decoder_path: str,
    input_path: str,
    output_path: str,
    secret: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
    device: Optional[str] = None,
    return_residual: bool = False,
) -> None:
    result = embed_image(encoder_path, decoder_path, input_path, secret, device=device, return_residual=return_residual)
    if return_residual:
        encoded_pil, residual_pil = result  # type: ignore
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        encoded_pil.save(output_path)
        base, ext = os.path.splitext(output_path)
        residual_pil.save(base + "_residual.png")
    else:
        encoded_pil = result  # type: ignore
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        encoded_pil.save(output_path)


def decode_file(
    encoder_path: str,
    decoder_path: str,
    input_path: str,
    device: Optional[str] = None,
    return_bits: bool = False,
) -> Union[str, np.ndarray, Tuple[Optional[str], np.ndarray]]:
    return decode_image(encoder_path, decoder_path, input_path, device=device, return_bits=return_bits)
