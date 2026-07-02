# DWSF Weight Provenance

- Algorithm: DWSF, Practical Deep Dispersed Watermarking with Synchronization and Fusion.
- Official source repository: https://github.com/bytedance/DWSF
- Official source commit reproduced in scratch: `2b3a4dbf5239a43d69cb4e5d090d4c856f7b6b6a`
- Paper: https://dl.acm.org/doi/10.1145/3581783.3612015
- arXiv: https://arxiv.org/abs/2310.14532
- Packaged files:
  - `encoder_best.pth`
  - `decoder_best.pth`
  - `seg.pth`
- Weight bundle source: WIBE DWSF wrapper download URL, `https://nextcloud.ispras.ru/index.php/s/F39nKXowAEpZyMy/download`
- WIBE repository: https://github.com/ispras/wibe
- WIBE DWSF submodule commit used for the path-aware segmentation smoke test: `63c0ec745d55054f3850346961f956410b41d32b`

The official DWSF repository does not publish a GitHub release or bundled
pretrained checkpoints. The three packaged checkpoint files were obtained from
the WIBE benchmark's DWSF model bundle and verified locally before promotion.
