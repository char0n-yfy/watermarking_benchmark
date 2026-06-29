"""3D viewpoint re-rendering attacks."""

from . import attacks as _attacks
from .attacks import (
    DEFAULT_MAX_DISPARITY_LEVELS,
    DEFAULT_SHARP_LOOKAT_MODES,
    DEFAULT_SHARP_PHASES,
    VIEWPOINT_ATTACK_CLASSES,
    ViewpointRerendering3DVariantAttack,
)

for _cls in VIEWPOINT_ATTACK_CLASSES:
    globals()[_cls.__name__] = _cls

__all__ = [
    "DEFAULT_MAX_DISPARITY_LEVELS",
    "DEFAULT_SHARP_LOOKAT_MODES",
    "DEFAULT_SHARP_PHASES",
    "VIEWPOINT_ATTACK_CLASSES",
    "ViewpointRerendering3DVariantAttack",
    *[cls.__name__ for cls in VIEWPOINT_ATTACK_CLASSES],
]

del _attacks
del _cls
