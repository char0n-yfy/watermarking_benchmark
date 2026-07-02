const WATERMARK_DISPLAY_NAMES: Record<string, string> = {
  "invisible-watermark-dwtdct": "DWT-DCT",
  "invisible-watermark-dwtdctsvd": "DWT-DCT-SVD",
  "invisible-watermark-rivagan": "RivaGAN",
  "traditional-spread-dct": "DCT",
};

export function normalizeWatermarkMethod(methodOrId: string | undefined): string {
  return (methodOrId ?? "").replace(/^alg-/, "");
}

export function resolveWatermarkDisplayName(methodOrId: string | undefined, fallback: string): string {
  const method = normalizeWatermarkMethod(methodOrId);
  return WATERMARK_DISPLAY_NAMES[method] ?? fallback;
}

export function watermarkMethodSubtitle(methodOrId: string | undefined): string {
  const method = normalizeWatermarkMethod(methodOrId);
  if (method.startsWith("invisible-watermark-")) {
    return method.slice("invisible-watermark-".length);
  }
  if (method.startsWith("traditional-spread-")) {
    return method.slice("traditional-spread-".length);
  }
  return method;
}
