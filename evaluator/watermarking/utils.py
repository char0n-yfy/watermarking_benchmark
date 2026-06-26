from __future__ import annotations

import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS_ROOT = PACKAGE_ROOT / "algorithms"
PACKAGED_WEIGHTS_ROOT = PACKAGE_ROOT / "weights" / "watermarking"
RESOURCE_WEIGHTS_ROOT = PROJECT_ROOT / "resources" / "weights" / "watermarking"


def packaged_algorithm_dir(name: str) -> Path:
    return ALGORITHMS_ROOT / name


def packaged_weights_dir(name: str) -> Path:
    resource_path = RESOURCE_WEIGHTS_ROOT / name
    if resource_path.exists():
        return resource_path
    package_path = PACKAGED_WEIGHTS_ROOT / name
    if package_path.exists():
        return package_path
    return resource_path


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_path(value: str | Path | None, label: str) -> Path:
    path = resolve_path(value)
    if path is None:
        raise ValueError(f"Missing required path: {label}")
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def normalize_device(device: str | None) -> str:
    if not device:
        return "cuda" if _cuda_available() else "cpu"
    if device.startswith("cuda") and not _cuda_available():
        return "cpu"
    return device


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def bits_from_message(message: str | None, nbits: int, *, seed: int | None = None) -> list[int]:
    if nbits <= 0:
        raise ValueError("nbits must be positive")

    if message and set(message) <= {"0", "1"} and len(message) == nbits:
        return [int(bit) for bit in message]

    if message:
        raw_bits: list[int] = []
        for byte in message.encode("utf-8"):
            raw_bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
        if raw_bits:
            repeats = (nbits + len(raw_bits) - 1) // len(raw_bits)
            return (raw_bits * repeats)[:nbits]

    rng = random.Random(seed)
    return [rng.randint(0, 1) for _ in range(nbits)]


def bits_to_string(bits: Iterable[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def bit_accuracy(expected: Iterable[int], decoded: Iterable[int]) -> float:
    expected_list = [int(bit) for bit in expected]
    decoded_list = [int(bit) for bit in decoded]
    n = min(len(expected_list), len(decoded_list))
    if n == 0:
        return 0.0
    return sum(a == b for a, b in zip(expected_list[:n], decoded_list[:n])) / n


def bits_to_numpy(bits: Iterable[int]) -> np.ndarray:
    return np.asarray([int(bit) for bit in bits], dtype=np.uint8)


@contextmanager
def prepend_sys_path(path: Path, purge_modules: Iterable[str] = ()) -> Iterator[None]:
    path = path.resolve()
    original_path = list(sys.path)
    for module_name in purge_modules:
        _purge_module_tree(module_name)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original_path


def _purge_module_tree(module_name: str) -> None:
    to_delete = [
        name
        for name in sys.modules
        if name == module_name or name.startswith(module_name + ".")
    ]
    for name in to_delete:
        sys.modules.pop(name, None)
