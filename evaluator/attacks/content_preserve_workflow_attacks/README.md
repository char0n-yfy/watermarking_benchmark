# Content-Preserving Workflow Attacks

本目录实现内容保持型数字工作流攻击。它们不同于单个低级 `distortion` 算子，而是模拟真实用户、修图软件和内容平台对图像进行修复、增强、导出、再处理的流程。

接口仍然保持图片文件输入、图片文件输出：

```text
watermarked image path -> workflow-attacked image path
```

## 已实现攻击

Restoration:

- `cp_denoise`
- `cp_deblock`
- `cp_deartifact`
- `cp_edge_preserve_smooth`

Resolution workflow:

- `cp_super_resolution`
- `cp_thumbnail_restore`

Enhancement:

- `cp_auto_enhance`
- `cp_clahe`
- `cp_sharpen`
- `cp_color_balance`

Style / App editing:

- `cp_filter_lut`
- `cp_warm_cold_tone`
- `cp_vivid_boost`

Platform workflow:

- `cp_platform_pipeline`
- `cp_social_export`
- `cp_iterative_export`
- `cp_color_space_pipeline`

Composite retouch workflow:

- `cp_retouch_pipeline_core`
- `cp_app_edit_pipeline`
- `cp_platform_retouch`
- `cp_clean_export_pipeline`

## 调用示例

```python
from pathlib import Path
from evaluator.attacks.runner import AttackJob, run_attack_dir

run_attack_dir(AttackJob(
    run_id="run_001",
    attack_name="cp_retouch_pipeline_core",
    params={"denoise_strength": 0.45, "export_quality": 88},
    input_dir=Path("runs/run_001/watermarked"),
    output_dir=Path("runs/run_001/attacked/cp_retouch_medium"),
    seed=42,
))
```

## 实现说明

- Core 版不依赖深度权重，主要使用 PIL、NumPy，以及可选 OpenCV。
- 如果 OpenCV 可用，`cp_denoise` 会使用 NLM，`cp_clahe` 会使用 LAB 亮度通道 CLAHE，边缘保持平滑会使用 bilateral filter。
- 所有结果统一保存为 PNG，便于后续检测/提取阶段读取。
- Deep 版攻击后续可以继续放在本目录，仍然继承同一个 `BaseAttack`。
