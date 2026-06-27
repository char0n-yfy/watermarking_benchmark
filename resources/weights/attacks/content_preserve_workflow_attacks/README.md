# Content-Preserving Workflow Attack Weights

Place Deep attack weights here. Core attacks do not need files in this directory.

Default layout:

```text
resources/weights/attacks/content_preserve_workflow_attacks/
├── denoise/<model_name>/
├── deblock/<model_name>/
├── deartifact/<model_name>/
├── super_resolution/<model_name>/
├── thumbnail_restore/<model_name>/
└── restore_pipeline/<model_name>/
```

The current Deep attack classes use local PyTorch backends for Restormer,
SwinIR JPEG/CAR, and Real-ESRGAN RRDBNet x4. They record `backend`,
`weight_path`, `weight_exists`, `weight_files`, and `fallback_used` in attack
metadata. If `allow_fallback=True`, a model error falls back to a deterministic
Core workflow; set `allow_fallback=False` to require real Deep inference.
