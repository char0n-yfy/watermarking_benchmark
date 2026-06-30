#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path


def _checkpoint_dir() -> Path:
    try:
        import torch

        return Path(torch.hub.get_dir()) / "checkpoints"
    except Exception:
        return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def _load_weights() -> None:
    from torchvision.models import AlexNet_Weights, VGG16_Weights, alexnet, vgg16

    checkpoints = _checkpoint_dir()
    print(f"Preparing perceptual metric weights in: {checkpoints}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

        print("Preparing AlexNet weights for LPIPS...")
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        print("Preparing VGG16 weights for DISTS...")
        vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    print("Perceptual metric weights are ready.")


def main() -> int:
    _load_weights()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
