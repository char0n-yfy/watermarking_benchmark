from __future__ import annotations

from evaluator.watermarking.methods._videoseal_family import VideoSealFamilyWatermark
from evaluator.watermarking.registry import register_watermark


@register_watermark
class PixelSealWatermark(VideoSealFamilyWatermark):
    name = "pixelseal"
    description = "PixelSeal wrapper using packaged 256-bit checkpoint from the VideoSeal repository."
    algorithm_dir_name = "pixelseal"
    checkpoint_filename = "pixelseal_checkpoint.pth"
    default_payload_bits = 256
    display_name = "PixelSeal"
