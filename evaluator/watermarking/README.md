# Watermarking Interface

This package mirrors the `evaluator.attacks` shape for watermark methods.

- `base.py`: shared context/result dataclasses and `BaseWatermark`.
- `registry.py`: method registration and construction.
- `runner.py`: directory-level embed/extract helpers.
- `methods/`: wrappers for concrete algorithms.

The package is intended to be movable with its local algorithm shims and
actual checkpoint files:

```text
evaluator/watermarking/
|-- algorithms/
`-- methods/

resources/weights/watermarking/
```

Current registered methods:

- `hidden`
- `ssl-watermarking`
- `stegastamp`
