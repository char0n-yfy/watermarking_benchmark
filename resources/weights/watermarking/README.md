# 水印算法权重

本目录用于存放水印算法相关权重。请按算法名称放入对应子目录。

预期目录结构：

```text
resources/weights/watermarking/
|-- hidden/
|   |-- combined-noise--epoch-400.pyt
|   `-- options-and-config.pickle
|-- ssl_watermarking/
|   |-- dino_r50_plus.pth
|   |-- out2048_coco_orig.pth
|   `-- ssl_carrier_seed2026.pt
`-- stegastamp/
    |-- encoder_best_loss_0.005250_step_66185.pth
    `-- decoder_best_loss_0.005250_step_66185.pth
```

`evaluator.watermarking` 中的算法封装会从
`resources/weights/watermarking/<method>/` 读取这些权重文件。
