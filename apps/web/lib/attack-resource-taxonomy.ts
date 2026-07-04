export const VIEWPOINT_MOTION_ORDER = ["swipe", "shake", "rotate", "rotate_forward"] as const;

export type ViewpointMotion = (typeof VIEWPOINT_MOTION_ORDER)[number];

export const VIEWPOINT_MOTION_LABELS: Record<ViewpointMotion, { en: string; zh: string }> = {
  swipe: { en: "Swipe", zh: "横向扫动" },
  shake: { en: "Shake", zh: "抖动" },
  rotate: { en: "Rotate", zh: "环绕旋转" },
  rotate_forward: { en: "Rotate Forward", zh: "前向环绕" }
};

export const ATTACK_RESOURCE_DISPLAY_NAMES: Record<string, { en: string; zh: string }> = {
  brightness: { en: "Brightness", zh: "亮度调整" },
  contrast: { en: "Contrast", zh: "对比度调整" },
  gaussian_blur: { en: "Gaussian Blur", zh: "高斯模糊" },
  gaussian_noise: { en: "Gaussian Noise", zh: "高斯噪声" },
  jpeg: { en: "JPEG Compression", zh: "JPEG 压缩" },
  resize: { en: "Resize", zh: "缩放" },
  resized_crop: { en: "Resized Crop", zh: "缩放裁剪" },
  rotation: { en: "Rotation", zh: "旋转" },
  erasing: { en: "Random Erasing", zh: "区域擦除" },
  screen_shoot: { en: "PIMoG-style Screen-Camera", zh: "屏幕-拍摄信道" },
  print_camera: { en: "CamMark-style Print-Camera", zh: "打印-拍摄信道" },
  combined_physical: { en: "Combined Physical Channel", zh: "组合物理信道" },
  "2x_regen": { en: "2-pass Diffusion Regeneration", zh: "2轮扩散再生成" },
  "4x_regen": { en: "4-pass Diffusion Regeneration", zh: "4轮扩散再生成" },
  regen_diffusion: { en: "WAVES Diffusion Regeneration", zh: "扩散再生成" },
  noise_to_image: { en: "CtrlRegen Noise-to-Image", zh: "噪声到图像再生成" },
  regen_vae: { en: "CompressAI VAE Reconstruction", zh: "VAE 再生成" },
  image_to_vedio: { en: "NFPA Image-to-Video", zh: "图像到视频再生成" },
  cew_e1: { en: "Auto-Tone", zh: "自动色调" },
  cew_e2: { en: "Warm-Vivid", zh: "暖色鲜艳" },
  cew_e3: { en: "Film-Faded", zh: "胶片褪色" },
  cew_e4: { en: "Local-Clarity HDR", zh: "局部清晰 HDR" },
  cew_c1: { en: "Basic Auto-Fix SR", zh: "自动修复+超分" },
  cew_c2: { en: "Color Retouch SR", zh: "色彩修饰+超分" },
  cew_c3: { en: "Detail Enhance SR", zh: "细节增强+超分" },
  cew_c4: { en: "Full Enhancement Chain", zh: "完整增强链" },
  cew_d1: { en: "Zero-DCE++ Auto-Light", zh: "自动补光" },
  cew_d2: { en: "DeepWB Auto-WhiteBalance", zh: "自动白平衡" },
  cew_d3: { en: "Image-Adaptive 3D LUT", zh: "自适应 AI 色彩" },
  cew_d4: { en: "Retinexformer Detail Low-Light Enhance", zh: "低光细节增强" },
  cew_d5: { en: "NAFNet/Restormer AI-Denoise", zh: "AI 去噪" },
  cew_s1: { en: "Real-ESRGAN", zh: "Real-ESRGAN" },
  cew_s2: { en: "SwinIR", zh: "SwinIR" },
  cew_s3: { en: "BSRGAN", zh: "BSRGAN" }
};

export const ATTACK_RESOURCE_METHOD_ORDER: Record<string, number> = {
  brightness: 10,
  contrast: 11,
  gaussian_blur: 12,
  gaussian_noise: 13,
  jpeg: 14,
  resize: 15,
  resized_crop: 16,
  rotation: 17,
  erasing: 18,
  screen_shoot: 20,
  print_camera: 21,
  combined_physical: 22,
  "2x_regen": 40,
  "4x_regen": 41,
  regen_diffusion: 42,
  noise_to_image: 43,
  regen_vae: 44,
  image_to_vedio: 45,
  cew_e1: 50,
  cew_e2: 51,
  cew_e3: 52,
  cew_e4: 53,
  cew_c1: 54,
  cew_c2: 55,
  cew_c3: 56,
  cew_c4: 57,
  cew_d1: 58,
  cew_d2: 59,
  cew_d3: 60,
  cew_d4: 61,
  cew_d5: 62,
  cew_s1: 63,
  cew_s2: 64,
  cew_s3: 65
};

export const REGENERATION_DIFFUSION_FAMILY_METHODS = ["regen_diffusion", "2x_regen", "4x_regen"] as const;

export const REGENERATION_RESOURCE_METHODS = [
  "regen_diffusion",
  "2x_regen",
  "4x_regen",
  "noise_to_image",
  "regen_vae",
  "image_to_vedio"
] as const;

export type AttackResourceCategoryKey =
  | "distortion_attacks"
  | "physical_channel_attacks"
  | "3d_viewpoint_rerendering"
  | "regeneration_attacks"
  | "consumer_enhancement_workflow_attacks";

const VIEWPOINT_EXECUTION_PATTERN =
  /^3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_(point|ahead)$/;

export const RESOURCE_PAGE_ATTACK_METHODS: Record<AttackResourceCategoryKey, readonly string[]> = {
  distortion_attacks: [
    "brightness",
    "contrast",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg",
    "resize",
    "resized_crop",
    "rotation",
    "erasing"
  ],
  physical_channel_attacks: ["screen_shoot", "print_camera", "combined_physical"],
  "3d_viewpoint_rerendering": [...VIEWPOINT_MOTION_ORDER],
  regeneration_attacks: [...REGENERATION_RESOURCE_METHODS],
  consumer_enhancement_workflow_attacks: Object.keys(ATTACK_RESOURCE_METHOD_ORDER)
    .filter((method) => method.startsWith("cew_"))
    .sort((left, right) => ATTACK_RESOURCE_METHOD_ORDER[left] - ATTACK_RESOURCE_METHOD_ORDER[right])
};

export function attackResourceDisplayName(method: string, language: string): string {
  const display = ATTACK_RESOURCE_DISPLAY_NAMES[method];
  if (display) {
    return language === "zh" ? display.zh : display.en;
  }
  const motion = VIEWPOINT_MOTION_LABELS[method as ViewpointMotion];
  if (motion) {
    return language === "zh" ? motion.zh : motion.en;
  }
  return method;
}

export function resourceMethodFromExecutionMethod(attackMethod: string): string | null {
  const viewpoint = attackMethod.match(VIEWPOINT_EXECUTION_PATTERN);
  if (viewpoint) {
    return viewpoint[1] ?? null;
  }
  if (attackMethod in ATTACK_RESOURCE_DISPLAY_NAMES) {
    return attackMethod;
  }
  return null;
}

export function executionMethodMatchesResourceMethod(
  attackMethod: string,
  resourceMethod: string
): boolean {
  if (resourceMethod === attackMethod) {
    return true;
  }
  if ((VIEWPOINT_MOTION_ORDER as readonly string[]).includes(resourceMethod)) {
    if (resourceMethod === "rotate") {
      return (
        attackMethod.startsWith("3d_viewpoint_rerendering_rotate_") &&
        !attackMethod.startsWith("3d_viewpoint_rerendering_rotate_forward_")
      );
    }
    return attackMethod.startsWith(`3d_viewpoint_rerendering_${resourceMethod}_`);
  }
  return false;
}

export type ResourcePageDetailAxisSpec = {
  key: string;
  labelZh: string;
  labelEn: string;
  order: number;
  matchMethod: (method: string) => boolean;
};

export function buildResourcePageDetailAxes(
  categoryKey: AttackResourceCategoryKey
): ResourcePageDetailAxisSpec[] {
  return RESOURCE_PAGE_ATTACK_METHODS[categoryKey].map((resourceMethod) => {
    const order = ATTACK_RESOURCE_METHOD_ORDER[resourceMethod] ?? 100;
    const motionLabel = VIEWPOINT_MOTION_LABELS[resourceMethod as ViewpointMotion];
    const display = ATTACK_RESOURCE_DISPLAY_NAMES[resourceMethod];
    return {
      key: resourceMethod,
      labelZh: motionLabel?.zh ?? display?.zh ?? resourceMethod,
      labelEn: motionLabel?.en ?? display?.en ?? resourceMethod,
      order,
      matchMethod: (method) => executionMethodMatchesResourceMethod(method, resourceMethod)
    };
  });
}
