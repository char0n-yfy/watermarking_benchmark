# Generative Attacks

本目录预留给生成式攻击，例如 VAE reconstruction、diffusion regeneration、image editing、rinse 等。

接口仍然必须遵守顶层约定：

```text
input watermarked image path -> output attacked image path
```

模型权重、prompt、scheduler、step、guidance、strength 等运行参数放在 attack `params` 或 `AttackContext.extra` 中。
