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

再生成攻击模型资源放在：

```text
resources/weights/attacks/regeneration_attacks/
├── diffusion/sd2-1-base/       # Stable Diffusion 2.1-base 资源目录
├── noise_to_image/             # CtrlRegen 资源目录
│   ├── adapters/ctrlregen/     # yepengliu/ctrlregen adapter snapshot
│   ├── base_model/Realistic_Vision_V4.0_noVAE/
│   ├── image_encoder/dinov2-giant/
│   └── vae/sd-vae-ft-mse/
└── vae/<vae_model_name>/*.pth.tar
```

3D viewpoint re-rendering 已分离为独立攻击类，权重也使用独立目录：

```text
resources/weights/attacks/3d_viewpoint_rerendering/
└── checkpoints/sharp_2572gikvuh.pt
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

## 3D viewpoint re-rendering

3D viewpoint re-rendering 已从本 regeneration 包拆到
`evaluator/attacks/3d_viewpoint_rerendering/`，权重也使用独立目录：

```text
resources/weights/attacks/3d_viewpoint_rerendering/
└── checkpoints/sharp_2572gikvuh.pt
```

完整方法名、强度映射和运行要求见
`evaluator/attacks/3d_viewpoint_rerendering/README.md`。

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

- `step=1.0`，前端强度参数名也是 `step`。`step` 会线性映射到
  `noise_step=100..1000`，即 `step=0` 对应 `noise_step=100`，
  `step=1` 对应 full-strength `noise_step=1000`。
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
