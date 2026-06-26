# 权重资源目录

本目录用于存放 benchmark 中各模块使用的本地模型权重。

大型二进制权重文件不提交到 Git。目录中保留 README 文件，用于让
GitHub 保存预期的目录结构，并说明每个目录应该放哪些权重。

预期目录结构：

```text
resources/weights/
|-- attacks/
`-- watermarking/
    |-- hidden/
    |-- ssl_watermarking/
    `-- stegastamp/
```

如果后续新增了其他权重文件格式，请先更新 `.gitignore`，再把文件放入本目录。
