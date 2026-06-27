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

权重统一放在：

```text
resources/weights/attacks/regeneration_attacks/
├── diffusion/sd2-1-base/       # Stable Diffusion 2.1-base 资源目录
└── vae/<vae_model_name>/*.pth.tar
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
