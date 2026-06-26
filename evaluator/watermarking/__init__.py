from .base import BaseWatermark, WatermarkContext, WatermarkEmbedResult, WatermarkExtractResult
from .methods import HiDDeNWatermark, SSLWatermark, StegaStampWatermark
from .registry import WATERMARK_REGISTRY, build_watermark, list_watermarks, register_watermark

__all__ = [
    "BaseWatermark",
    "HiDDeNWatermark",
    "SSLWatermark",
    "StegaStampWatermark",
    "WATERMARK_REGISTRY",
    "WatermarkContext",
    "WatermarkEmbedResult",
    "WatermarkExtractResult",
    "build_watermark",
    "list_watermarks",
    "register_watermark",
]
