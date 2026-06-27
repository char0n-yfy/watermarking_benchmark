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
- `trustmark`
- `trustmark-c`
- `trustmark-q`
- `rawatermark`
- `maskwm-d32`
- `videoseal`
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
such as PyTorch, torchvision, Pillow, NumPy, OpenCV and OmegaConf.
