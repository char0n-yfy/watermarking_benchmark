# Adversarial Attacks

本目录预留给对抗攻击，例如 feature-space PGD、surrogate detector attack、white-box removal/spoofing 等。

接口仍然必须遵守顶层约定：

```text
input watermarked image path -> output attacked image path
```

攻击内部可以使用 PyTorch、CLIP、VAE、surrogate detector 等模型，但不能把 tensor 或模型对象暴露给后续模块。
