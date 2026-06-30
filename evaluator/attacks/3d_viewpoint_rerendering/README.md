# 3D Viewpoint Re-rendering Attacks

本目录实现已从 `regeneration_attacks` 拆分出来的 3D viewpoint re-rendering
攻击。当前实现对应实验定义 `REG-3D-SHARP`，基于 Apple SHARP
(`apple/ml-sharp`)：先从单张水印图像预测 3D Gaussian Splatting 表示，再用
SHARP/gsplat 渲染新视角 PNG。

接口保持图片文件输入、图片文件输出：

```text
watermarked image path -> re-rendered attacked image path
```

## 资源目录

3D viewpoint re-rendering 的权重不再放在 regeneration 权重目录下，默认独立
读取：

```text
resources/weights/attacks/3d_viewpoint_rerendering/
└── checkpoints/sharp_2572gikvuh.pt
```

为了兼容旧 checkout，代码在默认新路径缺失时会回退查找旧路径：

```text
resources/weights/attacks/regeneration_attacks/3d_viewpoint_rerendering/
└── checkpoints/sharp_2572gikvuh.pt
```

新下载或新部署应只使用独立目录。`allow_download=True` 且 checkpoint 缺失时，
会从 Apple 官方模型 URL 下载到新的独立资源目录。

SHARP 源码 backend 仍复用现有 vendor 目录：

```text
evaluator/attacks/regeneration_attacks/backends/ml_sharp/
```

## 注册方法

4 种运动、8 个相位和 2 种 `lookat_mode` 被拆分为 64 个独立攻击方法：

```text
3d_viewpoint_rerendering_swipe_phase0_point
...
3d_viewpoint_rerendering_rotate_forward_phase7_point
3d_viewpoint_rerendering_swipe_phase0_ahead
...
3d_viewpoint_rerendering_rotate_forward_phase7_ahead
```

每个方法固定一个 `(motion, phase, lookat_mode)`，只输出当前固定组合的一张
PNG。前端资源页按 4 个 motion 方法展示；实验配置页再由 motion、phase 和
lookat_mode 三个独立维度展开到底层执行 preset。

为了兼容旧配置，旧 ID：

```text
3d_viewpoint_rerendering_phaseN_{point|ahead}
```

会映射到：

```text
3d_viewpoint_rerendering_rotate_phaseN_{point|ahead}
```

## 强度映射

前端和 API 统一使用 `strength in [0, 1]`，内部映射为：

```text
strength=0.0 -> max_disparity=0.01
strength=0.5 -> max_disparity=0.02
strength=1.0 -> max_disparity=0.04
```

中间值按 mild -> medium -> strong 分段线性插值。每次运行会在 metadata 中记录
`motion`、`trajectory_type`、`phase_index`、`phase`、`lookat_mode`、`strength`
和实际 `max_disparity`。

## 运行要求

SHARP 的 `gsplat` 渲染路径需要 CUDA GPU。本地 macOS/CPU 环境只能完成资源、
注册和接口校验，不能实际跑该攻击推理。

CUDA worker 环境需要安装 SHARP 依赖；建议执行：

```bash
python -m pip install -r evaluator/attacks/regeneration_attacks/backends/ml_sharp/requirements.txt
```

示例：

```python
from pathlib import Path
from evaluator.attacks.runner import AttackJob, run_attack_dir

run_attack_dir(AttackJob(
    run_id="sharp-viewpoint-demo",
    attack_name="3d_viewpoint_rerendering_rotate_phase0_point",
    params={"strength": 0.5},
    input_dir=Path("input_images"),
    output_dir=Path("attacked_images"),
    device="cuda",
))
```
