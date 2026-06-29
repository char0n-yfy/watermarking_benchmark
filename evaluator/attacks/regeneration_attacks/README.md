# Regeneration Attacks

本目录实现再生成攻击，例如 VAE reconstruction、diffusion regeneration、multi-step rinsing 等。

接口必须保持图片文件输入、图片文件输出：

```text
watermarked image path -> regenerated attacked image path
```

如果攻击产生 latent、prompt、mask 等辅助产物，应写入当前 attack 的工作目录，并在 `AttackResult.metadata` 中记录路径。

## `regen_vae`

`regen_vae` 通过 CompressAI VAE 模型重建水印图像：

```python
from evaluator.attacks.runner import AttackJob, run_attack_dir

run_attack_dir(AttackJob(
    run_id="regen-vae-demo",
    attack_name="regen_vae",
    params={
        "vae_model_name": "cheng2020-anchor",
        "quality": 3,
        "image_size": 512,
    },
    input_dir=Path("input_images"),
    output_dir=Path("attacked_images"),
    device="cuda",
))
```

可配置的 `vae_model_name`：

- `bmshj2018-factorized`
- `cheng2020-anchor`
- `bmshj2018-hyperprior`
- `mbt2018-mean`

模型资源统一放在：

```text
resources/weights/attacks/regeneration_attacks/
├── 3d_viewpoint_rerendering/
│   └── checkpoints/sharp_2572gikvuh.pt
├── diffusion/sd2-1-base/       # Stable Diffusion 2.1-base 资源目录
├── noise_to_image/             # CtrlRegen 资源目录
│   ├── adapters/ctrlregen/     # yepengliu/ctrlregen adapter snapshot
│   ├── base_model/Realistic_Vision_V4.0_noVAE/
│   ├── image_encoder/dinov2-giant/
│   └── vae/sd-vae-ft-mse/
└── vae/<vae_model_name>/*.pth.tar
```

第三方 backend/source 自包含在本代码包中，不放入权重目录：

```text
evaluator/attacks/regeneration_attacks/backends/
├── ctrlregen/
├── nfpa/
└── ml_sharp/
```

`regen_vae` 默认优先读取本地 `vae/<vae_model_name>/` 下的 CompressAI MSE checkpoint；缺失时按当前请求的 `vae_model_name` 和 `quality` 从 CompressAI 官方模型 URL 下载到该目录。

## `regen_diffusion`, `2x_regen`, `4x_regen`

这些攻击使用本地 Stable Diffusion 2.1-base Diffusers 权重：

```text
resources/weights/attacks/regeneration_attacks/diffusion/sd2-1-base/
```

默认超参数对齐 WatermarkAttacker/WAVES 的 Diffusion regeneration：

- `noise_step=60`
- `num_inference_steps=50`
- `guidance_scale=7.5`
- `image_size=512`
- `prompt=""`
- `negative_prompt=None`
- `seed=1024`
- `eta=0.0`

`head_start_step` 默认自动计算。对于默认 50-step 采样，它等价于参考实现中的：

```python
50 - max(noise_step // 20, 1)
```

因此默认 `noise_step=60` 时，`head_start_step=47`。`2x_regen` 和 `4x_regen` 分别连续执行 2 次和 4 次同样的 `regen_diffusion`，每一轮默认使用相同的 `noise_step` 和相同的 `seed`。

示例：

```python
from pathlib import Path
from evaluator.attacks.runner import AttackJob, run_attack_dir

run_attack_dir(AttackJob(
    run_id="regen-diffusion-demo",
    attack_name="4x_regen",
    params={
        "noise_step": 60,
        "num_inference_steps": 50,
        "image_size": 512,
    },
    input_dir=Path("input_images"),
    output_dir=Path("attacked_images"),
    device="cuda",
))
```

当前 Anaconda 环境中，直接导入 Diffusers 可能触发 `transformers -> TensorFlow/Keras -> pandas/pyarrow` 的底层段错误。本实现会在内部加载 Diffusers 前设置 `USE_TF=0` 和 `TRANSFORMERS_NO_TF=1`，使攻击路径保持在 PyTorch/Diffusers。

## `3d_viewpoint_rerendering`

`3d_viewpoint_rerendering` 是平台内接口名；实验定义名为
`REG-3D-SHARP-Rotate`。它基于 Apple SHARP (`apple/ml-sharp`) 实现
3D viewpoint re-rendering attack，先从单张水印图像预测 3D Gaussian
Splatting 表示，再用 SHARP/gsplat 渲染新视角 PNG。

默认资源目录：

```text
resources/weights/attacks/regeneration_attacks/3d_viewpoint_rerendering/
└── checkpoints/sharp_2572gikvuh.pt
```

SHARP 源码自包含在：

```text
evaluator/attacks/regeneration_attacks/backends/ml_sharp/
```

其中 `checkpoints/sharp_2572gikvuh.pt` 是 SHARP 官方 checkpoint。缺失且
`allow_download=True` 时，checkpoint 会从 Apple 官方模型 URL 下载到上述资源目录。
运行时会自动把 `backends/ml_sharp/src` 加入 Python 导入路径，但 CUDA worker
环境仍需要安装 SHARP 依赖；官方依赖至少包括 `torch`、`torchvision`、
`gsplat`、`timm`、`plyfile`、`scipy`、`imageio[ffmpeg]` 和 `pillow-heif`。
在 AutoDL 上建议进入 worker 环境后执行：

```bash
python -m pip install -r evaluator/attacks/regeneration_attacks/backends/ml_sharp/requirements.txt
```

SHARP 的 `gsplat` 渲染路径需要 CUDA GPU；本地 macOS/CPU 环境只能完成资源和
接口校验，不能实际跑该攻击推理。

攻击定义固定为：

- `trajectory_type="rotate"`
- `max_zoom=0.0`
- `phases=[0/8, 1/8, ..., 7/8]`
- `lookat_modes=["point", "ahead"]`
- 输出 PNG
- 攻击强度只由 `max_disparity` 控制

默认前端强度档位：

```text
0.01, 0.02, 0.04
```

每个 `max_disparity` 会生成 16 个子视角配置：

```text
8 phases x 2 lookat_modes
```

这些子输出会写入当前 run 的 `_intermediates/3d_viewpoint_rerendering/`
目录，并记录在 `AttackResult.metadata.variant_outputs` 中。受当前平台
“一张输入图 -> 一张攻击图”接口限制，主 `output_path` 会保存一个确定性代表
PNG；如果要严格复现主表里的 SHARP 均值定义，后续 scoring 应基于
`variant_outputs` 对 16 个子配置分别解码并取平均。

示例：

```python
run_attack_dir(AttackJob(
    run_id="sharp-viewpoint-demo",
    attack_name="3d_viewpoint_rerendering",
    params={"max_disparity": 0.02},
    input_dir=Path("input_images"),
    output_dir=Path("attacked_images"),
    device="cuda",
))
```

## `noise_to_image`

`noise_to_image` 是 CtrlRegen 的平台接口，默认把所有模型和 adapter 放在：

```text
resources/weights/attacks/regeneration_attacks/noise_to_image/
```

运行时还需要 CtrlRegen 官方源码中的 `custom_ip_adapter.py` 和
`custom_i2i_pipeline.py`，默认位置是：

```text
evaluator/attacks/regeneration_attacks/backends/ctrlregen/
```

默认参数：

- `step=1.0`，前端强度参数名也是 `step`。
- `num_inference_steps=50`
- `guidance_scale=2.0`
- `controlnet_conditioning_scale=1.0`
- `image_size=512`

本接口默认使用这些 Hugging Face 资源，并在本地缺失且
`allow_download=True` 时下载到上述资源目录：

- `SG161222/Realistic_Vision_V4.0_noVAE`
- `yepengliu/ctrlregen`
- `facebook/dinov2-giant`
- `stabilityai/sd-vae-ft-mse`

示例：

```python
run_attack_dir(AttackJob(
    run_id="ctrlregen-demo",
    attack_name="noise_to_image",
    params={"step": 0.75},
    input_dir=Path("input_images"),
    output_dir=Path("attacked_images"),
    device="cuda",
))
```

## `image_to_vedio`

`image_to_vedio` 是 NFPA 的平台接口，保留接口名中的 `vedio` 拼写以匹配
当前实验配置。它复用本目录下的 Stable Diffusion 2.1-base 权重：

```text
resources/weights/attacks/regeneration_attacks/diffusion/sd2-1-base/
```

运行时还需要 NFPA 官方源码中的 `utils.py`，默认位置是：

```text
evaluator/attacks/regeneration_attacks/backends/nfpa/utils.py
```

默认参数：

- `xy=40`，前端强度参数名也是 `xy`。
- `num_inference_steps=10`
- `image_size=512`
- `seed=1234`

示例：

```python
run_attack_dir(AttackJob(
    run_id="nfpa-demo",
    attack_name="image_to_vedio",
    params={"xy": 40},
    input_dir=Path("input_images"),
    output_dir=Path("attacked_images"),
    device="cuda",
))
```
