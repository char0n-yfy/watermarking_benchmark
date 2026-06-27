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
- `cp_despeckle`
- `cp_edge_preserve_smooth`
- `cp_denoise_deep`
- `cp_deblock_deep`
- `cp_deartifact_deep`

Resolution workflow:

- `cp_super_resolution`
- `cp_thumbnail_restore`
- `cp_resample_restore`
- `cp_sr_denoise`
- `cp_super_resolution_deep`
- `cp_thumbnail_restore_deep`

Enhancement:

- `cp_auto_enhance`
- `cp_clahe`
- `cp_hdr_like`
- `cp_sharpen`
- `cp_color_balance`

Style / App editing:

- `cp_filter_lut`
- `cp_warm_cold_tone`
- `cp_fade_matte`
- `cp_vivid_boost`
- `cp_mono_style`

Platform workflow:

- `cp_platform_pipeline`
- `cp_social_export`
- `cp_iterative_export`
- `cp_color_space_pipeline`
- `cp_metadata_strip_export`
- `cp_preview_pipeline`

Composite retouch workflow:

- `cp_retouch_pipeline_core`
- `cp_restore_pipeline_deep`
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
- Deep 版攻击已接入本地 PyTorch 推理 backend，默认读取 `resources/weights/attacks/content_preserve_workflow_attacks/` 下的对应子目录。
- 当前 Deep backend 包括 Restormer denoise、SwinIR JPEG/CAR 和 Real-ESRGAN RRDBNet x4；不依赖 `basicsr` 或 `realesrgan` 包。
- 如果权重缺失或推理失败，且 `allow_fallback=True`，会回退到可复现 Core fallback，并在 metadata 中记录 `fallback_used=True` 和 `fallback_reason`。
- 如果希望确认真实 Deep 推理生效，可以传入 `allow_fallback=False`；此时任何权重或模型错误都会让该样本失败。

## Deep 权重目录约定

默认根目录：

```text
resources/weights/attacks/content_preserve_workflow_attacks/
```

默认查找路径：

```text
denoise/<model_name>/
deblock/<model_name>/
deartifact/<model_name>/
super_resolution/<model_name>/
thumbnail_restore/<model_name>/
restore_pipeline/<model_name>/
```

也可以在攻击参数中传入 `weight_path` 指向单个权重文件或目录，或传入 `weight_root` 覆盖默认根目录。示例：

```python
run_attack_dir(AttackJob(
    run_id="run_001",
    attack_name="cp_super_resolution_deep",
    params={
        "model_name": "real_esrgan",
        "weight_path": "resources/weights/attacks/content_preserve_workflow_attacks/super_resolution/real_esrgan",
    },
    input_dir=Path("runs/run_001/watermarked"),
    output_dir=Path("runs/run_001/attacked/cp_sr_deep"),
))
```

运行 Deep 攻击需要：

```text
torch
einops
Pillow
NumPy
```
