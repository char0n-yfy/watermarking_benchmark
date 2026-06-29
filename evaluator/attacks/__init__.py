import importlib

from .base import AttackContext, AttackResult, BaseAttack
from . import consumer_enhancement_workflow_attacks as consumer_enhancement_workflow_attacks
from . import physical_channel_attacks as physical_channel_attacks
from . import regeneration_attacks as regeneration_attacks
viewpoint_rerendering_attacks = importlib.import_module(".3d_viewpoint_rerendering", __name__)
ViewpointRerendering3DVariantAttack = viewpoint_rerendering_attacks.ViewpointRerendering3DVariantAttack
from .distortion_attacks import (
    BrightnessAttack,
    ContrastAttack,
    ErasingAttack,
    GaussianBlurAttack,
    GaussianNoiseAttack,
    IdentityAttack,
    JPEGCompressionAttack,
    ResizedCropAttack,
    ResizeAttack,
    RotationAttack,
)
from .physical_channel_attacks import (
    CombinedPhysicalAttack,
    PrintCameraAttack,
    ScreenShootAttack,
)
from .registry import ATTACK_REGISTRY, build_attack, list_attacks, register_attack
from .regeneration_attacks import (
    FourTimesRegenDiffusionAttack,
    ImageToVedioAttack,
    NoiseToImageAttack,
    RegenDiffusionAttack,
    RegenVAEAttack,
    TwoTimesRegenDiffusionAttack,
)

__all__ = [
    "ATTACK_REGISTRY",
    "AttackContext",
    "AttackResult",
    "BaseAttack",
    "BrightnessAttack",
    "CombinedPhysicalAttack",
    "consumer_enhancement_workflow_attacks",
    "ContrastAttack",
    "ErasingAttack",
    "FourTimesRegenDiffusionAttack",
    "GaussianBlurAttack",
    "GaussianNoiseAttack",
    "ImageToVedioAttack",
    "IdentityAttack",
    "JPEGCompressionAttack",
    "NoiseToImageAttack",
    "physical_channel_attacks",
    "PrintCameraAttack",
    "RegenDiffusionAttack",
    "RegenVAEAttack",
    "ResizedCropAttack",
    "ResizeAttack",
    "RotationAttack",
    "ScreenShootAttack",
    "TwoTimesRegenDiffusionAttack",
    "ViewpointRerendering3DVariantAttack",
    "build_attack",
    "list_attacks",
    "register_attack",
    "regeneration_attacks",
    "viewpoint_rerendering_attacks",
]
