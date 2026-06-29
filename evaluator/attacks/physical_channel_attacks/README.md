# Physical Channel Attacks

This attack family contains the promoted v2 physical-channel simulation from
the scratch reproduction under `算法/attack/physical_channel_v2/`.

Implemented registry names:

- `screen_shoot`
- `print_camera`
- `combined_physical`

Design:

- `screen_shoot`: five-step PIMoG-style screen-camera chain:
  perspective, illumination, moire, camera imaging, JPEG.
- `print_camera`: five-step CamMark-style print-camera chain:
  print rendering, perspective, illumination, camera imaging, JPEG.
- `combined_physical`: two-hop `print_camera -> screen_shoot` chain with
  reduced strength per hop.

Strength control:

- All three registry names accept `strength` in `[0, 1]`.
- `0.0`, `0.5`, and `1.0` map to the original `mild`, `medium`, and
  `strong` operating points for `screen_shoot` and `print_camera`.
- `combined_physical` maps the global strength to the two-hop chain with the
  same reduced-hop rule as the old presets: `mild=(mild, mild)`,
  `medium=(medium, mild)`, and `strong=(medium, medium)`.

The promoted presets use the less-blur version selected after visual review of
the 12 uploaded real photos. The current defocus values are:

- screen: `0.35 / 0.65 / 0.95`
- print: `0.45 / 1.30 / 1.65`

Smoke-test references are recorded in:

```text
算法/logs/physical_channel_v2_less_blur.log
算法/outputs/physical_channel_v2_less_blur_preview/
算法/outputs/physical_channel_v2_less_blur_redline/
算法/outputs/physical_channel_v2_less_blur_heatmaps/
```
