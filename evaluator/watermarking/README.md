# Watermarking Interface

This package mirrors the `evaluator.attacks` shape for watermark methods.

- `base.py`: shared context/result dataclasses and `BaseWatermark`.
- `registry.py`: method registration and construction.
- `runner.py`: directory-level embed/extract helpers.
- `methods/`: wrappers for concrete algorithms.

The package follows the original project layout: production algorithm files
live under `evaluator/watermarking`, while actual checkpoint files live under
the root-level `resources/weights/watermarking` tree.

```text
evaluator/watermarking/
|-- algorithms/<method>/
`-- methods/<method>.py

resources/weights/watermarking/<method>/
```

Current registered methods:

- `traditional-dct`
- `traditional-spread-dct`
- `traditional-lsb`
- `traditional-haar`
- `blind_watermark`
- `invisible-watermark-dwtdct`
- `invisible-watermark-dwtdctsvd`
- `invisible-watermark-rivagan`
- `trustmark`
- `trustmark-c`
- `trustmark-q`
- `rawatermark`
- `maskwm-d32`
- `videoseal`
- `wam`
- `mbrs`
- `cin`
- `pimog`
- `hidden`
- `ssl-watermarking`
- `stegastamp`

Packaging convention:

- Prototype or reproduce a method under the top-level algorithm scratch area first.
- Move production-ready source files into `evaluator/watermarking/algorithms/<method>/`.
- Move actual checkpoint/model files into `resources/weights/watermarking/<method>/`.
- Add a wrapper in `evaluator/watermarking/methods/<method>.py` that only reads packaged algorithm/weight paths.

The fast methods added in this pass are self-contained under `evaluator/watermarking`
plus `resources/weights/watermarking`, except for normal Python runtime dependencies
such as PyTorch, torchvision, Pillow, NumPy, OpenCV, Kornia and OmegaConf.

Newly promoted GitHub methods:

- `wam`: Watermark Anything 32-bit localized watermark, using `wam_mit.pth`.
- `mbrs`: MBRS 256-bit JPEG-robust baseline, using `EC_42.pth`.
- `cin`: CIN 30-bit combined-noise baseline, using `cinNet_nsmNet.pth`.
- `pimog`: PIMoG 30-bit ScreenShooting baseline, using `Encoder_Decoder_Model_mask_99.pth`.

`pixelseal` and `chunkyseal` remain scratch-only references because their
checkpoints are too large for the current formal benchmark package. `invismark`
is also scratch-only until its official `paper.ckpt` can be downloaded without
interactive OneDrive authentication.
