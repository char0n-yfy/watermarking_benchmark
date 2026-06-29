"""
physical_channel_attacks.py  (v2 — 短链版 + 组合信道)
================================================================================
物理信道攻击模拟器：屏摄 (screen-shooting) / 打印翻拍 (print-camera) / 组合 (combined)。

设计原则（本版相对 v1 的改动）：
  1) 链条砍短，每步对应一个有文献依据的物理失真，便于解释：
     - screen_shoot 依据 PIMoG (ACM MM'22)：屏摄最核心失真 = 透视 + 光照 + 摩尔纹，
       其余用噪声兜底。PIMoG 原话："只放最主要的失真就够了"。
       => 5 步：透视 → 光照 → 摩尔纹 → 相机成像(轻模糊+噪声) → JPEG
     - print_camera 依据 CamMark (TOMM'15)：翻拍伪影 = 几何(透视/镜头) + 重采样/子采样
       + 色彩/亮度(AGC/AWB)；打印另加 半色调 + CMYK 色域。
       => 5 步：打印渲染(CMYK+半色调) → 透视 → 光照 → 相机成像(轻模糊+噪声+AGC/AWB色彩) → JPEG
     - 两条链共享一个"相机拍摄核心"(透视 + 光照 + 成像 + JPEG)，只差一处：
       屏摄多"摩尔纹"，翻拍前置"打印渲染"。这是唯一结构差异，最易解释。
  2) 去掉 v1 里造成绿色色偏/损毁的激进白平衡(wam wb_r/wb_b)与暗角；强度整体调温和，
     保证即使 strong 也是"失真不损毁"，语义可辨。
  3) 新增 combined：将"打印翻拍 → 屏摄"两跳串联，模拟真实世界跨媒介多跳传播
     (如：打印→拍照→上网→截屏)。每跳强度降档，避免双跳叠加把图毁掉。
  4) 本正式版由 `算法/attack/physical_channel_v2` scratch 复现提升而来，已按
     physical_channel_v2_package/任务书_物理信道攻击调优与标定方法.md
     做过真实照片红线自检、文字可读性降模糊复核与 StegaStamp 小样本标定：
     screen medium 贴近 0/20/40 度 BER 锚点量级，print medium 进入 NSN 占位地板范围，
     combined 使用降档双跳，在红线内保持 BER 高于单跳。

接口对齐 evaluator.attacks.base.BaseAttack（图入图出、AttackContext、PNG、seed 确定性）。
依赖：numpy, opencv-python(cv2), Pillow（均为 CPU、亚秒）。
================================================================================
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

if __name__ == "__main__":
    class AttackContext:  # type: ignore
        def __init__(self, run_id, sample_id, attack_name, params=None,
                     workspace_dir=None, device="cpu", seed=None, extra=None):
            self.run_id = run_id
            self.sample_id = sample_id
            self.attack_name = attack_name
            self.params = params or {}
            self.workspace_dir = workspace_dir
            self.device = device
            self.seed = seed
            self.extra = extra or {}

    class BaseAttack:  # type: ignore
        name = "base"
        description = ""
        output_ext = ".png"

        def __init__(self, **params: Any) -> None:
            self.params = dict(params)

    def register_attack(cls):  # type: ignore
        return cls
else:
    from evaluator.attacks.base import AttackContext, BaseAttack
    from evaluator.attacks.registry import register_attack

import cv2  # noqa: E402


# =============================================================================
# 通用工具
# =============================================================================
def _rng(context: AttackContext) -> np.random.Generator:
    return np.random.default_rng(context.seed)


def _load_bgr(path: Path) -> np.ndarray:
    return cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def _save_png(bgr: np.ndarray, path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)).save(path, "PNG")


def _odd(k) -> int:
    k = int(round(k)); return k + 1 if k % 2 == 0 else max(1, k)


# =============================================================================
# 失真构件（building blocks）—— 每个对应一个物理来源
# =============================================================================
def euler_homography(w, h, yaw, pitch, roll, f_scale=1.2) -> np.ndarray:
    """相机欧拉角(度) → 单应矩阵 H，模拟相机光轴未对准被拍平面。"""
    yaw, pitch, roll = map(math.radians, (yaw, pitch, roll))
    f = f_scale * max(w, h)
    src = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]], np.float64)
    cy, sy = math.cos(yaw), math.sin(yaw); cx, sx = math.cos(pitch), math.sin(pitch); cz, sz = math.cos(roll), math.sin(roll)
    R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @
         np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
         np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
    dst = []
    for (x, y) in src:
        p = R @ np.array([x, y, 0.0]); z = p[2] + f; z = z if abs(z) > 1e-6 else 1e-6
        dst.append([p[0]*f/z + w/2, p[1]*f/z + h/2])
    H, _ = cv2.findHomography((src + np.array([w/2, h/2])).astype(np.float32), np.array(dst, np.float32))
    return H


def perspective(bgr, H):
    h, w = bgr.shape[:2]
    return cv2.warpPerspective(bgr, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def defocus_blur(bgr, sigma):
    if sigma <= 0: return bgr
    return cv2.GaussianBlur(bgr, (_odd(2*round(sigma)+1), _odd(2*round(sigma)+1)), float(sigma))


def illumination(bgr, amount, glare, rng):
    """PIMoG 光照不均：低频乘性光场 + 可选高光斑。"""
    if amount <= 0 and glare <= 0: return bgr
    h, w = bgr.shape[:2]; x = bgr.astype(np.float32)
    if amount > 0:
        field = cv2.resize(rng.normal(0, 1, (4, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        field = (field - field.min()) / (np.ptp(field) + 1e-6)
        x = x * (1.0 + amount * (field - 0.5) * 2.0)[..., None]
    if glare > 0:
        gy, gx = rng.uniform(0.2, 0.8)*h, rng.uniform(0.2, 0.8)*w
        yy, xx = np.mgrid[0:h, 0:w]; sigma = 0.18*max(h, w)
        spot = np.exp(-(((xx-gx)**2 + (yy-gy)**2)/(2*sigma**2)))
        x = x + glare*120.0*spot[..., None]
    return np.clip(x, 0, 255)


def moire(bgr, intensity, period_px, orient_deg, rng, localize=True):
    """PIMoG 摩尔纹（局部包络版）：多正弦光栅叠加 + RGB 相位差 → 彩色拍频；
    低频空间包络使其片状出现，而非全幅均匀。"""
    if intensity <= 0: return bgr
    h, w = bgr.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); out = bgr.astype(np.float32)
    if localize:
        env = cv2.resize(rng.normal(0, 1, (3, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        env = np.clip(((env-env.min())/(np.ptp(env)+1e-6))**2.0, 0, 1); env = 0.08 + 0.92*env
    else:
        env = np.ones((h, w), np.float32)
    base_f = 2*math.pi/max(2.0, period_px)
    for c in range(3):
        patt = np.zeros((h, w), np.float32)
        for g in range(2):
            ang = math.radians(orient_deg + rng.uniform(-8, 8) + g*1.5)
            f = base_f*(1.0 + g*0.03 + rng.uniform(-0.01, 0.01)); phase = c*(2*math.pi/3) + rng.uniform(0, 0.5)
            patt += np.sin(f*(xx*math.cos(ang) + yy*math.sin(ang)) + phase)
        out[..., c] = out[..., c]*(1.0 + intensity*env*(patt/2))
    return np.clip(out, 0, 255)


def sensor_noise(bgr, sigma, rng):
    if sigma <= 0: return bgr
    return np.clip(bgr.astype(np.float32) + rng.normal(0, sigma*255.0, bgr.shape).astype(np.float32), 0, 255)


def color_shift(bgr, gamma=1.0, brightness=1.0, contrast=1.0, sat=1.0):
    """温和的色彩/亮度变换：CamMark AGC(增益)+AWB 的近似。默认接近中性，避免色偏损毁。"""
    x = np.power(np.clip(bgr.astype(np.float32)/255.0, 0, 1), max(0.05, gamma))
    x = np.clip((x - 0.5)*contrast + 0.5, 0, 1)*brightness; x = np.clip(x, 0, 1)
    if abs(sat - 1.0) > 1e-6:
        hsv = cv2.cvtColor((x*255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1]*sat, 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)/255.0
    return np.clip(x, 0, 1)*255.0


def jpeg_roundtrip(bgr, quality):
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else bgr


# ---- 打印专用 ----
def halftone(bgr, cell):
    """半色调/dot-gain 近似：Bayer 有序抖动 + 网点融合 + 暗部加深。"""
    if cell <= 1: return bgr
    x = bgr.astype(np.float32)/255.0
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)/16.0
    h, w = x.shape[:2]; tile = np.tile(bayer, (h//4+1, w//4+1))[:h, :w]
    dots = np.stack([(x[..., c] > tile).astype(np.float32) for c in range(3)], -1)*255.0
    dots = cv2.GaussianBlur(dots, (_odd(cell), _odd(cell)), 0)*0.97
    strength = min(0.30, 0.06 + 0.045*float(cell))
    return np.clip((bgr.astype(np.float32)*(1.0 - strength)) + dots*strength, 0, 255)


def paper_texture(bgr, amount, rng):
    if amount <= 0: return bgr
    h, w = bgr.shape[:2]
    tex = cv2.GaussianBlur(rng.normal(0, amount*255.0, (h, w)).astype(np.float32), (3, 3), 0)
    return np.clip(bgr.astype(np.float32) + tex[..., None], 0, 255)


def cmyk_gamut(bgr, amount):
    """CMYK 色域压缩近似：降饱和 + 轻提亮（打印色域比屏幕窄）。"""
    return bgr if amount <= 0 else color_shift(bgr, sat=1.0 - 0.25*amount, brightness=1.0 + 0.03*amount)


# =============================================================================
# 强度预设（已调温和：strong 也保持语义可辨）
# 注意：screen 不含白平衡/暗角（v1 绿色色偏的根源已移除）；色彩保持近中性。
# =============================================================================
SCREEN_PRESETS: dict[str, dict[str, Any]] = {
    #         yaw pit rol  dist  dfoc  illum glare moire mperiod noise jpeg  gamma  bri   con
    "mild":   dict(yaw=5,  pitch=3,  roll=2,  distance=0.96, dfocus=0.35, illum=0.060, glare=0.00,
                   moire=0.080, moire_period=11, noise=0.014, jpeg=90, gamma=1.015, brightness=1.01, contrast=1.03),
    "medium": dict(yaw=12, pitch=7,  roll=5,  distance=0.84, dfocus=0.65, illum=0.120, glare=0.025,
                   moire=0.240, moire_period=8,  noise=0.032, jpeg=78, gamma=1.035, brightness=1.02, contrast=1.05),
    "strong": dict(yaw=22, pitch=13, roll=8, distance=0.70, dfocus=0.95, illum=0.180, glare=0.070,
                   moire=0.340, moire_period=6,  noise=0.040, jpeg=72, gamma=1.06, brightness=1.03, contrast=1.08),
}

PRINT_PRESETS: dict[str, dict[str, Any]] = {
    "mild":   dict(yaw=5,  pitch=3,  roll=2,  distance=0.94, dfocus=0.45, illum=0.070, glare=0.020,
                   halftone_cell=3, paper=0.010, cmyk=0.40, noise=0.015, jpeg=88,
                   gamma=1.020, brightness=1.01, contrast=1.035, sat=0.94),
    "medium": dict(yaw=12, pitch=7,  roll=4,  distance=0.84, dfocus=1.30, illum=0.130, glare=0.060,
                   halftone_cell=6, paper=0.020, cmyk=1.05, noise=0.030, jpeg=78,
                   gamma=1.060, brightness=1.03, contrast=1.080, sat=0.86),
    "strong": dict(yaw=21, pitch=12, roll=7,  distance=0.72, dfocus=1.65, illum=0.180, glare=0.080,
                   halftone_cell=7, paper=0.028, cmyk=1.20, noise=0.040, jpeg=72,
                   gamma=1.085, brightness=1.04, contrast=1.110, sat=0.82),
}

# combined：每档 = (print 跳强度, screen 跳强度)，降档避免双跳叠加损毁
COMBINED_PRESETS: dict[str, tuple[str, str]] = {
    "mild":   ("mild",   "mild"),
    "medium": ("medium", "mild"),
    "strong": ("medium", "medium"),
}

LEVEL_STRENGTHS = {
    "mild": 0.0,
    "medium": 0.5,
    "strong": 1.0,
}


def _merge(level, presets, over):
    base = dict(presets.get(level, presets["medium"]))
    base.update({k: v for k, v in over.items() if v is not None})
    return base


def _clamp_unit_strength(strength):
    return max(0.0, min(1.0, float(strength)))


def _interp_piecewise(strength, mild, medium, strong):
    value = _clamp_unit_strength(strength)
    if value <= 0.5:
        return mild + (medium - mild) * (value / 0.5)
    return medium + (strong - medium) * ((value - 0.5) / 0.5)


def _preset_for_strength(presets, strength):
    numeric_keys = {
        key
        for key in presets["medium"]
        if isinstance(presets["mild"].get(key), (int, float))
        and isinstance(presets["medium"].get(key), (int, float))
        and isinstance(presets["strong"].get(key), (int, float))
    }
    resolved = dict(presets["medium"])
    for key in numeric_keys:
        resolved[key] = _interp_piecewise(strength, presets["mild"][key], presets["medium"][key], presets["strong"][key])
    return resolved


def _merge_strength(level, strength, presets, over):
    if strength is None:
        resolved_strength = LEVEL_STRENGTHS.get(level, LEVEL_STRENGTHS["medium"])
    else:
        resolved_strength = _clamp_unit_strength(strength)
    base = _preset_for_strength(presets, resolved_strength)
    merged = dict(base)
    merged.update({k: v for k, v in over.items() if v is not None})
    return merged, resolved_strength, base


def _combined_hop_strengths(level, strength):
    if strength is None:
        resolved_strength = LEVEL_STRENGTHS.get(level, LEVEL_STRENGTHS["medium"])
    else:
        resolved_strength = _clamp_unit_strength(strength)
    if resolved_strength <= 0.5:
        print_strength = resolved_strength
        screen_strength = 0.0
    else:
        print_strength = 0.5
        screen_strength = resolved_strength - 0.5
    return resolved_strength, print_strength, screen_strength


# =============================================================================
# 组合攻击 1：屏摄 screen_shoot  （5 步，PIMoG 依据）
# 透视 → 光照 → 摩尔纹 → 相机成像(轻模糊+噪声+近中性色彩) → JPEG → (可选)透视校正
# =============================================================================
@register_attack
class ScreenShootAttack(BaseAttack):
    name = "screen_shoot"
    description = "Simulated screen-shooting (PIMoG-style: perspective+illumination+moire+imaging+jpeg)."

    def __init__(self, level="medium", strength=None, correct_perspective=True,
                 yaw=None, pitch=None, roll=None, distance=None, moire=None, **kw):
        super().__init__(level=level, strength=strength, correct_perspective=correct_perspective,
                         yaw=yaw, pitch=pitch, roll=roll, distance=distance, moire=moire, **kw)
        self.level, self.strength, self.correct = level, strength, correct_perspective
        self.over = dict(yaw=yaw, pitch=pitch, roll=roll, distance=distance, moire=moire, **kw)

    def apply(self, input_path, output_path, context) -> Mapping[str, Any]:
        rng = _rng(context); p, strength, _ = _merge_strength(self.level, self.strength, SCREEN_PRESETS, self.over)
        bgr = _load_bgr(Path(input_path)); h, w = bgr.shape[:2]
        H = euler_homography(w, h, p["yaw"], p["pitch"], p["roll"])
        x = perspective(bgr, H)                                                   # 1 透视
        x = illumination(x, p["illum"], p["glare"], rng)                          # 2 光照
        ang = 1.0 + max(abs(p["yaw"]), abs(p["pitch"]))/30.0                      #   角度耦合
        x = moire(x, p["moire"], p["moire_period"]/max(0.3, p["distance"]), p["roll"], rng)  # 3 摩尔纹(频率∼距离)
        x = defocus_blur(x, p["dfocus"]*ang)                                      # 4 成像:模糊(∼角度)
        x = sensor_noise(x, p["noise"], rng)                                      #     +噪声
        x = color_shift(x, p["gamma"], p["brightness"], p["contrast"])           #     +近中性色彩(AGC)
        x = jpeg_roundtrip(x, p["jpeg"])                                          # 5 JPEG
        corrected = False
        if self.correct:
            x = cv2.warpPerspective(x, np.linalg.inv(H), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            corrected = True
        _save_png(x, Path(output_path))
        return {"channel": "screen_shoot", "level": self.level, "strength": strength, "perspective_corrected": corrected,
                "yaw": p["yaw"], "pitch": p["pitch"], "roll": p["roll"],
                "distance": p["distance"], "moire": p["moire"], "jpeg": p["jpeg"]}


# =============================================================================
# 组合攻击 2：打印翻拍 print_camera  （5 步，CamMark 依据，无摩尔纹）
# 打印渲染(CMYK+半色调+纸纹) → 透视 → 光照 → 相机成像(模糊+噪声+AGC/AWB色彩) → JPEG → (可选)校正
# =============================================================================
@register_attack
class PrintCameraAttack(BaseAttack):
    name = "print_camera"
    description = "Simulated print-camera (CamMark-style capture + halftone/CMYK; no screen moire)."

    def __init__(self, level="medium", strength=None, correct_perspective=True,
                 yaw=None, pitch=None, roll=None, distance=None, **kw):
        super().__init__(level=level, strength=strength, correct_perspective=correct_perspective,
                         yaw=yaw, pitch=pitch, roll=roll, distance=distance, **kw)
        self.level, self.strength, self.correct = level, strength, correct_perspective
        self.over = dict(yaw=yaw, pitch=pitch, roll=roll, distance=distance, **kw)

    def apply(self, input_path, output_path, context) -> Mapping[str, Any]:
        rng = _rng(context); p, strength, base = _merge_strength(self.level, self.strength, PRINT_PRESETS, self.over)
        base_distance = float(base["distance"])
        distance_scale = float(np.clip(base_distance / max(0.3, float(p["distance"])), 0.65, 1.60))
        bgr = _load_bgr(Path(input_path)); h, w = bgr.shape[:2]
        x = cmyk_gamut(bgr, p["cmyk"]); x = halftone(x, p["halftone_cell"]); x = paper_texture(x, p["paper"], rng)  # 1 打印渲染
        H = euler_homography(w, h, p["yaw"], p["pitch"], p["roll"])
        x = perspective(x, H)                                                     # 2 透视
        x = illumination(x, p["illum"]*distance_scale, p["glare"]*distance_scale, rng)  # 3 光照(∼距离)
        ang = 1.0 + max(abs(p["yaw"]), abs(p["pitch"]))/30.0
        x = defocus_blur(x, p["dfocus"]*ang*distance_scale)
        x = sensor_noise(x, p["noise"]*distance_scale, rng)                       # 4 成像:模糊+噪声(∼距离)
        x = color_shift(x, p["gamma"], p["brightness"], p["contrast"], p["sat"])  #     +AGC/AWB色彩
        x = jpeg_roundtrip(x, p["jpeg"])                                          # 5 JPEG
        corrected = False
        if self.correct:
            x = cv2.warpPerspective(x, np.linalg.inv(H), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            corrected = True
        _save_png(x, Path(output_path))
        return {"channel": "print_camera", "level": self.level, "strength": strength, "perspective_corrected": corrected,
                "yaw": p["yaw"], "distance": p["distance"], "distance_scale": distance_scale,
                "halftone_cell": p["halftone_cell"], "cmyk": p["cmyk"], "jpeg": p["jpeg"]}


# =============================================================================
# 组合攻击 3：跨媒介组合 combined  （打印翻拍 → 屏摄，两跳串联）
# 模拟真实多跳：打印→拍照→上网→屏幕显示→截屏/再拍。每跳降档，避免叠加损毁。
# =============================================================================
@register_attack
class CombinedPhysicalAttack(BaseAttack):
    name = "combined_physical"
    description = "Cross-media chain: print_camera then screen_shoot, reduced strength per hop."

    def __init__(self, level="medium", strength=None, order="print_then_screen", correct_perspective=True,
                 yaw=None, pitch=None, roll=None, distance=None, **kw):
        super().__init__(level=level, strength=strength, order=order, correct_perspective=correct_perspective,
                         yaw=yaw, pitch=pitch, roll=roll, distance=distance, **kw)
        self.level, self.strength, self.order, self.correct = level, strength, order, correct_perspective
        self.over = dict(yaw=yaw, pitch=pitch, roll=roll, distance=distance, **kw)

    def apply(self, input_path, output_path, context) -> Mapping[str, Any]:
        strength, print_strength, screen_strength = _combined_hop_strengths(self.level, self.strength)
        work = Path(context.workspace_dir or "/tmp"); work.mkdir(parents=True, exist_ok=True)
        mid = work / f"_combined_mid_{context.sample_id}.png"
        hops = ([("print", PrintCameraAttack(strength=print_strength, correct_perspective=self.correct, **self.over)),
                 ("screen", ScreenShootAttack(strength=screen_strength, correct_perspective=self.correct, **self.over))]
                if self.order == "print_then_screen" else
                [("screen", ScreenShootAttack(strength=screen_strength, correct_perspective=self.correct, **self.over)),
                 ("print", PrintCameraAttack(strength=print_strength, correct_perspective=self.correct, **self.over))])
        cur, log = Path(input_path), []
        for i, (tag, atk) in enumerate(hops):
            nxt = mid if i == 0 else Path(output_path)
            ctx = AttackContext(run_id=context.run_id, sample_id=context.sample_id, attack_name=atk.name,
                                workspace_dir=work, device=context.device,
                                seed=None if context.seed is None else context.seed*100 + i)
            m = atk.apply(cur, nxt, ctx); log.append({tag: m}); cur = nxt
        return {"channel": "combined_physical", "level": self.level, "strength": strength, "order": self.order,
                "print_strength": print_strength, "screen_strength": screen_strength, "hops": log}


# =============================================================================
# 自测
# =============================================================================
if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    g = np.zeros((512, 512, 3), np.uint8); g[::4, :, :] = 255; g[:, ::4, :] = 255
    cv2.circle(g, (256, 256), 120, (0, 128, 255), -1); cv2.imwrite(str(tmp/"s.png"), g)
    for atk in (ScreenShootAttack(level="strong"), PrintCameraAttack(level="strong"),
                CombinedPhysicalAttack(level="strong")):
        ctx = AttackContext(
            run_id="physical_channel_selftest",
            sample_id="synthetic",
            attack_name=atk.name,
            params=atk.params,
            workspace_dir=tmp,
            device="cpu",
            seed=42,
            extra={},
        )
        out = tmp/f"{atk.name}.png"; meta = atk.apply(tmp/"s.png", out, ctx)
        assert Image.open(out).size == (512, 512)
        print(f"[OK] {atk.name:18s} size=512  channel={meta['channel']}")
    print("\nAll v2 physical-channel smoke tests passed.")
