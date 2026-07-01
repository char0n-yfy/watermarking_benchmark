# 水印算法简介

更新时间：2026-07-01

范围：本文件按当前 `evaluator.watermarking.list_watermarks()` 的注册顺序整理，共 21 个水印算法。`算法 ID` 是前端和后端可直接匹配的稳定键。

| 序号 | 算法 ID | 中文简介 | 论文地址 | 仓库/源码地址 |
| --- | --- | --- | --- | --- |
| 1 | `chunkyseal` | VideoSeal 系列的高容量水印方法。本项目封装的是 1024-bit checkpoint，适合测试大 payload 容量下的鲁棒性和视觉质量。 | [ChunkySeal, arXiv:2510.12812](https://arxiv.org/abs/2510.12812) | [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal) |
| 2 | `cin` | CIN 是结合可逆与不可逆机制的神经图像水印。本项目使用 30-bit 预训练权重，主要用于评估组合失真下的盲水印鲁棒性。 | [arXiv:2212.12678](https://arxiv.org/abs/2212.12678)<br>[ACM MM 页面](https://dl.acm.org/doi/abs/10.1145/3503161.3547950) | [rmpku/CIN](https://github.com/rmpku/CIN) |
| 3 | `dwsf` | DWSF 是深度分散水印方法，通过分散嵌入、同步定位和多块融合提升裁剪、缩放等场景下的恢复能力。本项目使用 30-bit encoder/decoder 和 segmentation checkpoint。 | [ACM DL DOI:10.1145/3581783.3612015](https://dl.acm.org/doi/10.1145/3581783.3612015)<br>[arXiv:2310.14532](https://arxiv.org/abs/2310.14532) | [bytedance/DWSF](https://github.com/bytedance/DWSF) |
| 4 | `hidden` | HiDDeN 是经典深度编码器-解码器水印基线。本项目使用 30-bit 配置和 JPEG 压缩训练 checkpoint。 | [arXiv:1807.09937](https://arxiv.org/abs/1807.09937) | [ando-khachatryan/HiDDeN](https://github.com/ando-khachatryan/HiDDeN)<br>原始实现：[jirenz/HiDDeN](https://github.com/jirenz/HiDDeN) |
| 5 | `invismark` | InvisMark 面向 AI 图像溯源水印。本项目使用用户补齐的 `paper.ckpt` 编码器/解码器权重，支持 100-bit 二进制 payload。 | [arXiv:2411.07795](https://arxiv.org/abs/2411.07795) | [microsoft/InvisMark](https://github.com/microsoft/InvisMark) |
| 6 | `invisible-watermark-dwtdct` | ShieldMnt invisible-watermark 中的经典频域盲水印变体，组合 DWT 和 DCT，不需要神经网络权重，速度快但抗强缩放/裁剪能力较弱。 | 上游未绑定单篇论文；属于传统 DWT-DCT 水印基线。 | [ShieldMnt/invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) |
| 7 | `invisible-watermark-dwtdctsvd` | ShieldMnt invisible-watermark 中的 DWT-DCT-SVD 变体，在频域嵌入中加入 SVD。本项目将它与 DWT-DCT 拆开注册。 | 上游未绑定单篇论文；属于传统 DWT-DCT-SVD 水印基线。 | [ShieldMnt/invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) |
| 8 | `invisible-watermark-rivagan` | ShieldMnt 包中的 RivaGAN 神经水印变体。本项目使用 packaged ONNX 编码器/解码器，payload 固定为 32 bit。 | [RivaGAN, arXiv:1909.01285](https://arxiv.org/abs/1909.01285) | [ShieldMnt/invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) |
| 9 | `maskwm-d32` | MaskWM 的全局 32-bit 水印封装，使用上游 `D_32bits` checkpoint。本地处理尺寸固定为 512x512。 | [arXiv:2504.12739](https://arxiv.org/abs/2504.12739) | [hurunyi/MaskWM](https://github.com/hurunyi/MaskWM) |
| 10 | `mbrs` | MBRS 是面向 JPEG 鲁棒性的深度水印方法。本项目使用 `EC_42` 权重和 256-bit payload，训练中考虑真实与模拟 JPEG 压缩。 | [arXiv:2108.08211](https://arxiv.org/abs/2108.08211) | [jzyustc/MBRS](https://github.com/jzyustc/MBRS) |
| 11 | `pimog` | PIMoG 是面向屏幕拍摄场景的神经水印。本项目使用 30-bit checkpoint，固定 128x128 本地处理。 | [ACM DOI:10.1145/3503161.3548049](https://doi.org/10.1145/3503161.3548049) | [FangHanNUS/PIMoG](https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw) |
| 12 | `pixelseal` | PixelSeal 属于 VideoSeal 系列，本项目使用 256-bit checkpoint。它作为大权重鲁棒水印方法纳入测试。 | [PixelSeal, arXiv:2512.16874](https://arxiv.org/abs/2512.16874) | [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal) |
| 13 | `rawatermark` | RAWatermark 是 zero-bit 检测型水印，输出水印存在概率/检测结果，而不是解码消息。 | [arXiv:2403.18774](https://arxiv.org/abs/2403.18774) | [jeremyxianx/RAWatermark](https://github.com/jeremyxianx/RAWatermark) |
| 14 | `ssl-watermarking` | Meta 的自监督特征空间水印。本项目默认使用 30-bit multi-bit 模式，通过 DINO/ResNet 特征空间优化嵌入。 | [arXiv:2112.09581](https://arxiv.org/abs/2112.09581) | [facebookresearch/ssl_watermarking](https://github.com/facebookresearch/ssl_watermarking) |
| 15 | `stegastamp` | StegaStamp 面向打印-拍摄恢复场景。本项目使用 PyTorch 权重，可嵌入 100-bit 原始 payload，也可通过上游 API 嵌入带 BCH/ECC 的短文本。 | [arXiv:1904.05343](https://arxiv.org/abs/1904.05343) | [tancik/StegaStamp](https://github.com/tancik/StegaStamp) |
| 16 | `traditional-spread-dct` | 项目内置的传统 DCT 扩频水印基线，使用两个 DCT 系数承载信息，主要用于速度快、可解释的频域对照实验。 | 基础参考：[Cox 等扩频水印，DOI:10.1109/83.650120](https://doi.org/10.1109/83.650120) | 项目源码：`evaluator/watermarking/methods/traditional.py` |
| 17 | `trustmark-c` | TrustMark-C 是紧凑型神经水印，保持 TrustMark 100-bit 消息格式，同时更偏向小模型和快速推理。 | [arXiv:2311.18297](https://arxiv.org/abs/2311.18297)<br>[ICCV 2025 PDF](https://collomosse.com/pubs/Bui-ICCV-2025.pdf) | [adobe/trustmark](https://github.com/adobe/trustmark) |
| 18 | `trustmark-q` | TrustMark-Q 是质量优先的 TrustMark 变体，同样使用 100-bit 消息格式，但模型更重，侧重图像质量。 | [arXiv:2311.18297](https://arxiv.org/abs/2311.18297)<br>[ICCV 2025 PDF](https://collomosse.com/pubs/Bui-ICCV-2025.pdf) | [adobe/trustmark](https://github.com/adobe/trustmark) |
| 19 | `videoseal` | VideoSeal v1.0 的 image-mode 水印。本项目使用 256-bit 图像 checkpoint，输出保持输入尺寸。 | [VideoSeal, arXiv:2412.09492](https://arxiv.org/abs/2412.09492) | [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal) |
| 20 | `vine` | VINE 是使用生成式先验增强鲁棒性的 100-bit 水印。本项目封装 VINE-R，并依赖本地 SD-Turbo 组件，主要用于评估图像编辑下的鲁棒性。 | [arXiv:2410.18775](https://arxiv.org/abs/2410.18775) | [Shilin-LU/VINE](https://github.com/Shilin-LU/VINE) |
| 21 | `wam` | Meta Watermark Anything 的局部化水印方法。本项目使用 MIT 权重，支持 32-bit 消息，并能输出水印区域检测/定位热图。 | [arXiv:2411.07231](https://arxiv.org/abs/2411.07231) | [facebookresearch/watermark-anything](https://github.com/facebookresearch/watermark-anything) |

## 归纳

- 当前水印注册表共 21 个算法，覆盖传统频域、深度编码器-解码器、物理世界鲁棒水印、局部化水印、zero-bit 检测、VideoSeal 系列、扩散先验水印和分散同步水印。
- 容量跨度较大：`chunkyseal` 为 1024 bit 高容量方法，`videoseal`、`pixelseal`、`mbrs` 为 256 bit，TrustMark/InvisMark/VINE/StegaStamp 为 100 bit，WAM/MaskWM/HiDDeN/CIN/PIMoG/DWSF/SSL 约为 30 到 32 bit，RAWatermark 为 zero-bit 检测。
- DWSF 新增后补齐了“分散嵌入 + 同步定位 + 融合解码”这一类深度水印，对裁剪、缩放和几何扰动评测很有价值。
- 前端建议直接使用 `算法 ID` 作为 key；没有被 `evaluator/watermarking/methods/__init__.py` 导入的历史源码不计入这份 21 项清单。
