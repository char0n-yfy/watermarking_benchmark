# Physical Channel Attacks

This attack family contains the promoted v2 physical-channel simulation from
the scratch reproduction under `算法/attack/physical_channel_v2/`.

Implemented registry names:

- `screen_shoot`
- `print_camera`
- `combined_physical`
- `screen_shoot_mild`, `screen_shoot_medium`, `screen_shoot_strong`
- `print_camera_mild`, `print_camera_medium`, `print_camera_strong`
- `combined_physical_mild`, `combined_physical_medium`, `combined_physical_strong`
- `screen_shoot_mild_uncorrected`, `screen_shoot_medium_uncorrected`, `screen_shoot_strong_uncorrected`
- `print_camera_mild_uncorrected`, `print_camera_medium_uncorrected`, `print_camera_strong_uncorrected`

Design:

- `screen_shoot`: five-step PIMoG-style screen-camera chain:
  perspective, illumination, moire, camera imaging, JPEG.
- `print_camera`: five-step CamMark-style print-camera chain:
  print rendering, perspective, illumination, camera imaging, JPEG.
- `combined_physical`: two-hop `print_camera -> screen_shoot` chain with
  reduced strength per hop.

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
