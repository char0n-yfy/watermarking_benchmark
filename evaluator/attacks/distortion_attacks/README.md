# Distortion Attacks

本目录放基础数字失真攻击。它们都是轻量的图片到图片映射，不依赖水印算法内部状态。

当前攻击：

| name | 参数 | 说明 |
| --- | --- | --- |
| `identity` | 无 | 不攻击，直接复制图片 |
| `rotation` | `angle` 或 `strength` | 旋转图片 |
| `resized_crop` | `scale` 或 `strength` | 随机裁剪后缩放回原尺寸 |
| `erasing` | `area_ratio` 或 `strength` | 随机擦除局部区域 |
| `brightness` | `factor` 或 `strength` | 亮度变化 |
| `contrast` | `factor` 或 `strength` | 对比度变化 |
| `gaussian_blur` | `radius` 或 `strength` | 高斯模糊 |
| `gaussian_noise` | `sigma` 或 `strength` | 高斯噪声 |
| `jpeg` | `quality` 或 `strength` | JPEG 压缩后转回 PNG |
| `resize` | `scale` | 缩放再恢复原尺寸 |

`strength` 默认是 `[0, 1]` 的相对强度，会映射到各攻击的实际参数范围。例如：

```python
AttackJob(
    run_id="run_001",
    attack_name="jpeg",
    params={"strength": 0.5},
    input_dir=...,
    output_dir=...,
)
```

也可以直接使用绝对参数：

```python
AttackJob(
    run_id="run_001",
    attack_name="rotation",
    params={"angle": 30},
    input_dir=...,
    output_dir=...,
)
```

所有攻击输出 PNG，保证后续检测/提取阶段只处理统一图片格式。
