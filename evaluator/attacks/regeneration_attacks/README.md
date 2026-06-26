# Regeneration Attacks

本目录预留给再生成攻击，例如 VAE reconstruction、diffusion regeneration、multi-step rinsing 等。

接口必须保持图片文件输入、图片文件输出：

```text
watermarked image path -> regenerated attacked image path
```

如果攻击产生 latent、prompt、mask 等辅助产物，应写入当前 attack 的工作目录，并在 `AttackResult.metadata` 中记录路径。
