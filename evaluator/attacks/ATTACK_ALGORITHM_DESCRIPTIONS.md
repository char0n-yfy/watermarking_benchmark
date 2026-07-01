# 攻击算法简介

更新时间：2026-07-01

范围：本文件按当前 `evaluator.attacks.list_attacks()` 的注册顺序整理，共 43 个攻击算法。组合攻击如果没有单独论文或仓库，就拆解为内部包含的子攻击，并列出子攻击对应的论文/源码来源。`攻击 ID` 是前端和后端可直接匹配的稳定键。

## 消费级增强工作流攻击

| 序号 | 攻击 ID | 中文简介 | 论文/资料地址 | 仓库/源码地址 |
| --- | --- | --- | --- | --- |
| 1 | `cew_e1` | CEW-E1 Auto-Tone，项目内置的自动影调编辑预设，主要包含自动对比度、曝光、鲜艳度和锐化调整。 | 无独立论文；属于项目内消费级编辑预设。 | 项目源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py` |
| 2 | `cew_e2` | CEW-E2 Warm-Vivid，项目内置的暖色鲜艳化编辑预设，主要改变色温、色调、饱和度、对比度和曲线。 | 无独立论文；属于项目内消费级编辑预设。 | 项目源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py` |
| 3 | `cew_e3` | CEW-E3 Film-Faded，项目内置的胶片褪色风格预设，主要降低对比度/饱和度、抬升黑场并软化局部对比。 | 无独立论文；属于项目内消费级编辑预设。 | 项目源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py` |
| 4 | `cew_e4` | CEW-E4 Local-Clarity HDR，项目内置的局部清晰度/HDR 风格预设，主要包含暗部提升、局部对比增强和锐化。 | 无独立论文；属于项目内消费级编辑预设。 | 项目源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py` |
| 5 | `cew_d1` | CEW-D1 Auto-Light，自动低光增强攻击；有权重时使用 Zero-DCE++ 风格后端，无权重时使用确定性增强 fallback。 | [Zero-DCE++, arXiv:2103.00860](https://arxiv.org/abs/2103.00860)<br>[Zero-DCE, arXiv:2001.06826](https://arxiv.org/abs/2001.06826) | [Li-Chongyi/Zero-DCE_extension](https://github.com/Li-Chongyi/Zero-DCE_extension)<br>项目后端：`evaluator/attacks/consumer_enhancement_workflow_attacks/backends/` |
| 6 | `cew_d2` | CEW-D2 Auto-WhiteBalance，自动白平衡攻击；有权重时使用 Deep White-Balance Editing，无权重时使用灰世界白平衡 fallback。 | [Deep White-Balance Editing, arXiv:2004.01354](https://arxiv.org/abs/2004.01354) | [mahmoudnafifi/Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance)<br>项目后端：`evaluator/attacks/consumer_enhancement_workflow_attacks/backends/` |
| 7 | `cew_d3` | CEW-D3 Adaptive AI Color，自适应 AI 调色攻击；有权重时使用 Image-Adaptive 3D LUT 风格后端，无权重时使用本地 LUT fallback。 | [Image-Adaptive 3D LUT, arXiv:2009.14468](https://arxiv.org/abs/2009.14468) | [HuiZeng/Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)<br>项目后端：`evaluator/attacks/consumer_enhancement_workflow_attacks/backends/` |
| 8 | `cew_d4` | CEW-D4 Detail Low-Light Enhance，低光细节增强攻击；有权重时使用 Retinexformer 风格后端，无权重时使用 HDR-like fallback。 | [Retinexformer, arXiv:2303.06705](https://arxiv.org/abs/2303.06705) | [caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)<br>项目后端：`evaluator/attacks/consumer_enhancement_workflow_attacks/backends/` |
| 9 | `cew_d5` | CEW-D5 AI-Denoise Clean，AI 降噪/复原攻击，参考 NAFNet 和 Restormer 的图像复原任务；无权重时使用本地确定性降噪 fallback。 | [NAFNet, arXiv:2204.04676](https://arxiv.org/abs/2204.04676)<br>[Restormer, arXiv:2111.09881](https://arxiv.org/abs/2111.09881) | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)<br>[swz30/Restormer](https://github.com/swz30/Restormer) |
| 10 | `cew_s1` | CEW-S1 RealESRGAN，超分辨率攻击，支持 x2/x4 scale；权重不可用时退化到本地 resize+锐化 fallback。 | [Real-ESRGAN, arXiv:2107.10833](https://arxiv.org/abs/2107.10833) | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)<br>项目后端：`evaluator/attacks/consumer_enhancement_workflow_attacks/backends/` |
| 11 | `cew_s2` | CEW-S2 SwinIR，基于 Swin Transformer 思路的超分/复原攻击，支持 x2/x4 scale；权重不可用时使用本地 fallback。 | [SwinIR, arXiv:2108.10257](https://arxiv.org/abs/2108.10257) | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)<br>项目后端：`evaluator/attacks/consumer_enhancement_workflow_attacks/backends/` |
| 12 | `cew_s3` | CEW-S3 BSRGAN，盲超分辨率攻击，支持 x2/x4 scale；权重不可用时使用本地 fallback。 | [BSRGAN, arXiv:2103.14006](https://arxiv.org/abs/2103.14006) | [cszn/BSRGAN](https://github.com/cszn/BSRGAN)<br>项目后端：`evaluator/attacks/consumer_enhancement_workflow_attacks/backends/` |
| 13 | `cew_c1` | CEW-C1 Basic Auto-Fix SR，组合攻击：先执行 D1 自动亮度增强，再执行 D2 白平衡、D5 降噪，最后执行 S1 RealESRGAN 超分。 | 子攻击来源：[Zero-DCE++](https://arxiv.org/abs/2103.00860)、[DeepWB](https://arxiv.org/abs/2004.01354)、[NAFNet](https://arxiv.org/abs/2204.04676)、[Real-ESRGAN](https://arxiv.org/abs/2107.10833)。 | 组合源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py`<br>子攻击仓库：[Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension)、[DeepWB](https://github.com/mahmoudnafifi/Deep_White_Balance)、[NAFNet](https://github.com/megvii-research/NAFNet)、[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| 14 | `cew_c2` | CEW-C2 Color Retouch SR，组合攻击：先执行 D2 白平衡，再执行 D3 自适应调色、D5 降噪，最后执行 S2 SwinIR x2 超分。 | 子攻击来源：[DeepWB](https://arxiv.org/abs/2004.01354)、[Image-Adaptive 3D LUT](https://arxiv.org/abs/2009.14468)、[NAFNet](https://arxiv.org/abs/2204.04676)、[SwinIR](https://arxiv.org/abs/2108.10257)。 | 组合源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py`<br>子攻击仓库：[DeepWB](https://github.com/mahmoudnafifi/Deep_White_Balance)、[Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)、[NAFNet](https://github.com/megvii-research/NAFNet)、[SwinIR](https://github.com/JingyunLiang/SwinIR) |
| 15 | `cew_c3` | CEW-C3 Detail Enhance SR，组合攻击：先执行 D1 自动亮度增强，再执行 D4 低光细节增强、D5 降噪，最后执行 S1 RealESRGAN x4 超分。 | 子攻击来源：[Zero-DCE++](https://arxiv.org/abs/2103.00860)、[Retinexformer](https://arxiv.org/abs/2303.06705)、[NAFNet](https://arxiv.org/abs/2204.04676)、[Real-ESRGAN](https://arxiv.org/abs/2107.10833)。 | 组合源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py`<br>子攻击仓库：[Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension)、[Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)、[NAFNet](https://github.com/megvii-research/NAFNet)、[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| 16 | `cew_c4` | CEW-C4 Full Enhancement Chain，组合攻击：依次执行 D1 自动亮度、D2 白平衡、D3 自适应调色、D4 低光细节增强、D5 降噪，最后执行 S3 BSRGAN x4 超分。 | 子攻击来源：[Zero-DCE++](https://arxiv.org/abs/2103.00860)、[DeepWB](https://arxiv.org/abs/2004.01354)、[Image-Adaptive 3D LUT](https://arxiv.org/abs/2009.14468)、[Retinexformer](https://arxiv.org/abs/2303.06705)、[NAFNet](https://arxiv.org/abs/2204.04676)、[BSRGAN](https://arxiv.org/abs/2103.14006)。 | 组合源码：`evaluator/attacks/consumer_enhancement_workflow_attacks/attacks.py`<br>子攻击仓库：[Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension)、[DeepWB](https://github.com/mahmoudnafifi/Deep_White_Balance)、[Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)、[Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)、[NAFNet](https://github.com/megvii-research/NAFNet)、[BSRGAN](https://github.com/cszn/BSRGAN) |

## 物理信道攻击

| 序号 | 攻击 ID | 中文简介 | 论文/资料地址 | 仓库/源码地址 |
| --- | --- | --- | --- | --- |
| 17 | `screen_shoot` | 屏幕拍摄模拟攻击，组合透视变化、非均匀光照、摩尔纹、相机成像噪声/模糊和 JPEG 压缩，整体参考 PIMoG 的屏摄失真建模。 | [PIMoG, DOI:10.1145/3503161.3548049](https://doi.org/10.1145/3503161.3548049) | [FangHanNUS/PIMoG](https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw)<br>项目源码：`evaluator/attacks/physical_channel_attacks/attacks.py` |
| 18 | `print_camera` | 打印-翻拍模拟攻击，组合打印半色调/CMYK 色域、透视变化、光照扰动、相机成像和 JPEG 压缩，整体参考 CamMark 的相机链路思想。 | [CamMark, DOI:10.1145/2557642.2557644](https://doi.org/10.1145/2557642.2557644) | 未使用官方攻击仓库；项目源码：`evaluator/attacks/physical_channel_attacks/attacks.py` |
| 19 | `combined_physical` | 跨媒介组合物理攻击：先执行 `print_camera` 打印-翻拍链路，再执行 `screen_shoot` 屏摄链路，并降低每跳强度以模拟真实分享传播。 | 子攻击来源：[CamMark](https://doi.org/10.1145/2557642.2557644)、[PIMoG](https://doi.org/10.1145/3503161.3548049)。 | 组合源码：`evaluator/attacks/physical_channel_attacks/attacks.py`<br>子攻击来源：[PIMoG 仓库](https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw) 和项目内 CamMark-style 模拟 |

## 再生成攻击

| 序号 | 攻击 ID | 中文简介 | 论文/资料地址 | 仓库/源码地址 |
| --- | --- | --- | --- | --- |
| 20 | `regen_vae` | CompressAI VAE 重构攻击，把水印图像送入 learned image compression 模型再解码，借压缩重构破坏水印信号；支持 `bmshj2018-factorized`、`bmshj2018-hyperprior`、`mbt2018-mean`、`cheng2020-anchor` 等模型。 | [Ballé hyperprior, arXiv:1802.01436](https://arxiv.org/abs/1802.01436)<br>[Minnen et al., arXiv:1809.02736](https://arxiv.org/abs/1809.02736)<br>[Cheng2020, arXiv:2001.01568](https://arxiv.org/abs/2001.01568) | [InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI)<br>项目源码：`evaluator/attacks/regeneration_attacks/attacks.py` |
| 21 | `regen_diffusion` | 单次 Stable Diffusion 2.1-base 图像再生成攻击，通过 image-to-image diffusion 生成语义相近图像，从而削弱水印可恢复性。 | [Latent Diffusion Models, arXiv:2112.10752](https://arxiv.org/abs/2112.10752)<br>[WAVES benchmark, arXiv:2401.08573](https://arxiv.org/abs/2401.08573) | [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)<br>项目源码：`evaluator/attacks/regeneration_attacks/attacks.py` |
| 22 | `2x_regen` | 两次连续 Stable Diffusion 2.1-base 再生成攻击，本质是把 `regen_diffusion` 重复两轮，比单次扩散再生成更强。 | 组成来源：[Latent Diffusion Models](https://arxiv.org/abs/2112.10752)、[WAVES benchmark](https://arxiv.org/abs/2401.08573)。 | [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)<br>项目源码：`evaluator/attacks/regeneration_attacks/attacks.py` |
| 23 | `4x_regen` | 四次连续 Stable Diffusion 2.1-base 再生成攻击，本质是把 `regen_diffusion` 重复四轮，用于更强的水印移除压力测试。 | 组成来源：[Latent Diffusion Models](https://arxiv.org/abs/2112.10752)、[WAVES benchmark](https://arxiv.org/abs/2401.08573)。 | [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)<br>项目源码：`evaluator/attacks/regeneration_attacks/attacks.py` |
| 24 | `noise_to_image` | CtrlRegen noise-to-image 可控再生成攻击，从干净噪声出发并用条件控制生成视觉相关图像，以破坏原图中的水印信号。 | [CtrlRegen, arXiv:2410.05470](https://arxiv.org/abs/2410.05470) | [yepengliu/ctrlregen 模型资源](https://huggingface.co/yepengliu/ctrlregen)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ctrlregen/` |
| 25 | `image_to_vedio` | NFPA image-to-video next-frame prediction 攻击；项目 ID 保留 `vedio` 拼写。它使用 vendored Diffusers next-frame pipeline，把单图通过视频式下一帧预测再生成。 | 项目中未记录独立 NFPA 论文链接；底层扩散模型可参考 [Latent Diffusion Models, arXiv:2112.10752](https://arxiv.org/abs/2112.10752)。 | 项目后端：`evaluator/attacks/regeneration_attacks/backends/nfpa/` |

## 3D 视角重渲染攻击

本组 8 个方法属于同一个大攻击族：每个注册项固定一组 `(motion, lookat_mode)`，先用 SHARP 从单张水印图预测 3D Gaussian 表示，再从 8 个 phase 中随机抽取一个相位渲染新视角图像。旧的 phase 级攻击 ID 会兼容映射到对应的 motion/lookat 方法。

| 序号 | 攻击 ID | 中文简介 | 论文/资料地址 | 仓库/源码地址 |
| --- | --- | --- | --- | --- |
| 26 | `3d_viewpoint_rerendering_swipe_point` | REG-3D-SHARP 视角重渲染变体：swipe 运动，lookat 模式为 `point`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |
| 27 | `3d_viewpoint_rerendering_swipe_ahead` | REG-3D-SHARP 视角重渲染变体：swipe 运动，lookat 模式为 `ahead`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |
| 28 | `3d_viewpoint_rerendering_shake_point` | REG-3D-SHARP 视角重渲染变体：shake 运动，lookat 模式为 `point`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |
| 29 | `3d_viewpoint_rerendering_shake_ahead` | REG-3D-SHARP 视角重渲染变体：shake 运动，lookat 模式为 `ahead`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |
| 30 | `3d_viewpoint_rerendering_rotate_point` | REG-3D-SHARP 视角重渲染变体：rotate 运动，lookat 模式为 `point`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |
| 31 | `3d_viewpoint_rerendering_rotate_ahead` | REG-3D-SHARP 视角重渲染变体：rotate 运动，lookat 模式为 `ahead`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |
| 32 | `3d_viewpoint_rerendering_rotate_forward_point` | REG-3D-SHARP 视角重渲染变体：rotate_forward 运动，lookat 模式为 `point`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |
| 33 | `3d_viewpoint_rerendering_rotate_forward_ahead` | REG-3D-SHARP 视角重渲染变体：rotate_forward 运动，lookat 模式为 `ahead`；运行时从 8 个 phase 中随机抽取一个相位。 | [SHARP, arXiv:2512.10685](https://arxiv.org/abs/2512.10685) | [apple/ml-sharp](https://github.com/apple/ml-sharp)<br>项目后端：`evaluator/attacks/regeneration_attacks/backends/ml_sharp/` |

## 基础失真攻击

| 序号 | 攻击 ID | 中文简介 | 论文/资料地址 | 仓库/源码地址 |
| --- | --- | --- | --- | --- |
| 34 | `identity` | 无攻击基线，直接复制水印图像，用于衡量 clean 情况下的提取/检测性能。 | 无；benchmark baseline。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 35 | `rotation` | 几何旋转失真，按固定或相对角度旋转图像。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 36 | `resized_crop` | 裁剪再缩放失真，裁出方形区域后 resize 回原图尺寸。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 37 | `erasing` | 随机擦除/遮挡失真，在图像中抹除一个方形区域并保持原尺寸。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 38 | `brightness` | 亮度缩放失真，用于模拟简单曝光或亮度编辑。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 39 | `contrast` | 对比度缩放失真，用于模拟简单对比度编辑。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 40 | `gaussian_blur` | 高斯模糊失真，使用 Gaussian kernel 降低图像细节。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 41 | `gaussian_noise` | 加性零均值高斯噪声失真，用于模拟传感器或传输噪声。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 42 | `jpeg` | JPEG 压缩失真，内部执行 JPEG 编码/解码后再保存回 PNG，方便下游统一读取。 | [JPEG 标准 ISO/IEC 10918-1](https://www.iso.org/standard/18902.html) | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |
| 43 | `resize` | 缩放再恢复失真，先按比例 resize，再恢复到原图尺寸。 | 无；标准图像变换。 | 项目源码：`evaluator/attacks/distortion_attacks/attacks.py` |

## 归纳

- 当前攻击注册表共 43 个算法：16 个消费级增强工作流攻击、3 个物理信道攻击、6 个再生成攻击、8 个 3D 视角重渲染变体、10 个基础失真攻击。
- 组合攻击没有强行写成独立外部方法，而是拆开列出组成它的子攻击来源。例如 `cew_c1` 到 `cew_c4` 分别列出 D/S 子模块，`combined_physical` 列出 CamMark-style print-camera 与 PIMoG-style screen-shoot。
- 前端建议直接使用 `攻击 ID` 作为 key；若 `论文/资料地址` 写的是“无”或项目源码，说明该项是本 benchmark 的内部预设或标准失真变换，不是独立论文算法。
