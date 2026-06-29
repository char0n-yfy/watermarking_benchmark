# 攻击模块接口约定

本目录定义平台后端提供的攻击方法接口。设计原则是：

- 跨模块只交换图片文件，不传 tensor、PIL 对象或模型对象。
- 攻击模块输入是水印图片路径，输出是攻击后图片路径。
- 每个攻击必须写出图片结果，并返回一份 JSON 可序列化的元数据。
- 攻击内部可以使用 PIL、OpenCV、PyTorch、diffusers 等实现细节，但这些细节不能泄漏到外部接口。

## 标准数据流

```text
dataset image
  -> watermark embedder
  -> watermarked image file
  -> attack module
  -> attacked image file
  -> watermark detector/extractor
  -> metrics/report
```

## 单图接口

所有攻击类继承 `BaseAttack`，实现：

```python
def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
    ...
```

约定：

- `input_path`：水印图片文件。
- `output_path`：攻击后图片写入位置。
- `context`：任务 ID、样本 ID、随机种子、设备、工作目录等元数据。
- 返回值：JSON 可序列化字典，只放参数和诊断信息，不放图片内容。

推荐所有攻击最终输出 PNG，避免 JPEG 二次压缩或格式差异影响后续提取。如果某个攻击本身就是 JPEG 压缩，也先在内部做 JPEG 压缩，再转回 PNG 输出。

## 批量接口

后端调度时使用 `run_attack_dir`：

```python
from pathlib import Path
from evaluator.attacks.runner import AttackJob, run_attack_dir

job = AttackJob(
    run_id="run_001",
    attack_name="jpeg",
    params={"quality": 50},
    input_dir=Path("runs/run_001/watermarked"),
    output_dir=Path("runs/run_001/attacked/jpeg_50"),
    device="cuda:0",
    seed=42,
)

results = run_attack_dir(job)
```

输出目录示例：

```text
runs/run_001/attacked/jpeg_50/
├── 000001.png
├── 000002.png
└── attack_manifest.json
```

`attack_manifest.json` 记录每张图片的输入、输出、耗时、参数、错误信息等。

## 目录结构

```text
evaluator/attacks/
├── base.py                 # 通用数据结构和 BaseAttack
├── registry.py             # 攻击注册表
├── runner.py               # 目录级批量执行
├── distortion_attacks/     # 基础数字失真攻击
├── consumer_enhancement_workflow_attacks/ # 消费级增强工作流攻击
├── physical_channel_attacks/ # 物理信道模拟攻击
├── regeneration_attacks/   # 再生成攻击
└── adversarial_attacks/    # 对抗攻击预留
```

后续新增攻击优先按攻击族放进对应文件夹，不要都堆在顶层。

## Distortion Attacks

当前 `distortion_attacks/` 已实现：

- `rotation`
- `resized_crop`
- `erasing`
- `brightness`
- `contrast`
- `gaussian_blur`
- `gaussian_noise`
- `jpeg`
- `resize`

`identity` 是无攻击基线，当前在前端资源分类中单独归为 `identity`，不并入经典失真。

查询攻击：

```python
from evaluator.attacks import list_attacks

print(list_attacks())
```

## Physical Channel Attacks

`physical_channel_attacks/` 实现 v2 物理信道复现模拟，已从
`算法/attack/physical_channel_v2/` scratch 区提升到正式攻击注册表：

- `screen_shoot`：屏摄链路，包含透视、光照、摩尔纹、相机成像和 JPEG。
- `print_camera`：打印翻拍链路，包含打印渲染、透视、光照、相机成像和 JPEG。
- `combined_physical`：打印翻拍到屏摄的跨媒介双跳链路。

三类攻击现在都通过 `strength in [0, 1]` 控制强度，`0 / 0.5 / 1`
分别对应原 `mild / medium / strong` 档位。当前正式预设使用用户视觉复核后的
降模糊版本，避免文字类图片被过度模糊。

## 新增攻击

新增文件，例如 `evaluator/attacks/my_attack.py`：

```python
from pathlib import Path
from typing import Any, Mapping
from PIL import Image

from evaluator.attacks.base import AttackContext, BaseAttack
from evaluator.attacks.registry import register_attack


@register_attack
class MyAttack(BaseAttack):
    name = "my_attack"
    description = "Example image-to-image attack."

    def __init__(self, strength: float = 0.5) -> None:
        super().__init__(strength=strength)
        self.strength = strength

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        with Image.open(input_path) as image:
            attacked = image.convert("RGB")
            # do something
            attacked.save(output_path, format="PNG")
        return {"strength": self.strength}
```

然后在包初始化或后端启动时导入该模块，注册表就能发现它。

## 生成式攻击约定

生成式攻击仍然遵守“图片进、图片出”：

- 输入：水印图片路径。
- 输出：再生成后的图片路径。
- 模型路径、prompt、step、strength、guidance 等都放在 `params` 或 `context.extra`。
- 如果需要额外产物，例如 prompt、latent、attention map，写到 `context.workspace_dir` 下，并在返回 metadata 中记录路径。
