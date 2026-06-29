CEW backend implementations live here so the attack package keeps source code
separate from downloadable model resources.

- `deep_enhance.py`: local torch backends for DeepWB AWB, Image-Adaptive 3D LUT,
  and RetinexFormer low-light enhancement.
- `restoration_sr.py`: local torch backends for Zero-DCE++, Restormer denoise,
  RRDBNet/Real-ESRGAN/BSRGAN, and SwinIR super-resolution.

Model checkpoints stay under
`resources/weights/attacks/consumer_enhancement_workflow_attacks/`.

