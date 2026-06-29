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

- `invisible-watermark-dwtdct`
- `invisible-watermark-dwtdctsvd`
- `invisible-watermark-rivagan`
- `trustmark`
- `trustmark-c`
- `trustmark-q`
- `rawatermark`
- `maskwm-d32`
- `videoseal`
- `pixelseal`
- `chunkyseal`
- `vine`
- `wam`
- `mbrs`
- `cin`
- `pimog`
- `invismark`
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
- `pixelseal`: PixelSeal 256-bit VideoSeal-family checkpoint, using `pixelseal_checkpoint.pth`.
- `chunkyseal`: ChunkySeal 1024-bit high-capacity VideoSeal-family checkpoint, using `chunkyseal_checkpoint.pth`.
- `invismark`: InvisMark 100-bit AI provenance checkpoint, using `paper.ckpt`.
- `vine`: VINE 100-bit diffusion-prior watermark wrapper, using local VINE-R and SD-Turbo paths under `resources/weights/watermarking/vine/`.

`pixelseal` and `chunkyseal` are required-list size exceptions. `pixelseal`
still satisfies the warm-loaded subsecond gate locally; `chunkyseal` embeds
locally and decodes locally, but its 1024-bit model is not a strict fast method
on the 8 GB RTX 5070 Laptop GPU used for packaging. `invismark` was promoted
after the user supplied `ckpt_paper.zip`; the wrapper loads only the packaged
encoder/decoder checkpoint and does not instantiate the training discriminator
or LPIPS path.

`vine` is a requested diffusion-dependent registration exception. The wrapper
is formally registered and uses only local packaged paths:
`resources/weights/watermarking/vine/vine-r-dec/`,
`resources/weights/watermarking/vine/vine-r-enc/`, and
`resources/weights/watermarking/vine/sd-turbo/`. Full packaged-path
embed/extract smoke passed on CUDA with 100/100 decoded bits. The SD-Turbo
fp16 files are stored locally, while UNet and VAE are cast to fp32 at runtime
to avoid fp16 NaNs during VINE inference.
