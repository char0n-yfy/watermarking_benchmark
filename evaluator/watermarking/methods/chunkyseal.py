from __future__ import annotations

from evaluator.watermarking.methods._videoseal_family import VideoSealFamilyWatermark
from evaluator.watermarking.registry import register_watermark


@register_watermark
class ChunkySealWatermark(VideoSealFamilyWatermark):
    name = "chunkyseal"
    description = "ChunkySeal wrapper using packaged 1024-bit high-capacity checkpoint from the VideoSeal repository."
    algorithm_dir_name = "chunkyseal"
    checkpoint_filename = "chunkyseal_checkpoint.pth"
    default_payload_bits = 1024
    display_name = "ChunkySeal"
