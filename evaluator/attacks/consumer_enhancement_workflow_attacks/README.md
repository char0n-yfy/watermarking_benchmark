# Consumer Enhancement Workflow Attacks

This package implements the CEW-Bench v4 attack family described by the
benchmark spec:

- `cew_e1` ... `cew_e4`: 4 darktable-style edit attacks. Select
  `light`, `medium`, or `strong` with the `strength` parameter.
- `cew_d1` ... `cew_d5`: 5 deep-enhancement entries for auto-light,
  white balance, adaptive color, low-light detail enhancement, and AI denoise.
- `cew_s1` ... `cew_s3`: 3 super-resolution entries for Real-ESRGAN,
  SwinIR, and BSRGAN. Select x2 or x4 with the `scale` parameter.
- `cew_c1` ... `cew_c4`: 4 composite multi-step enhancement chains.

All outputs are saved as PNG and the implementation avoids JPEG recompression.

The deep entries first look under:

```text
resources/weights/attacks/consumer_enhancement_workflow_attacks/
```

The local torch backend implementations are kept under:

```text
evaluator/attacks/consumer_enhancement_workflow_attacks/backends/
├── deep_enhance.py
└── restoration_sr.py
```

`cew_d1`, `cew_d2`, `cew_d3`, `cew_d4`, `cew_d5`, and `cew_s*` all have local
torch inference backends. When a checkpoint is unavailable or cannot be loaded,
`allow_fallback=True` uses deterministic Pillow/NumPy/OpenCV-style image
operations and records `fallback_used=True` in the attack metadata. Set
`allow_fallback=False` when validating that a real checkpoint-backed path is
available.
