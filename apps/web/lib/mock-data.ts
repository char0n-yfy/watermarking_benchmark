import type { AlgorithmVersion, AttackPreset, DatasetVersion, ModelArtifact, RunStatus } from "./types";

export const datasets: DatasetVersion[] = [
  { id: "local-root", name: "Local dataset root", sampleCount: 1, version: "local" },
  { id: "ds-demo-v1", name: "Demo smoke set", sampleCount: 32, version: "v1" }
];

export const algorithms: AlgorithmVersion[] = [
  {
    id: "alg-invisible-watermark-dwtdct",
    name: "Invisible Watermark DWT-DCT",
    version: "packaged",
    status: "enabled",
    requiresGpu: false,
    method: "invisible-watermark-dwtdct",
    recommended: true
  },
  {
    id: "alg-invisible-watermark-dwtdctsvd",
    name: "Invisible Watermark DWT-DCT-SVD",
    version: "packaged",
    status: "enabled",
    requiresGpu: false,
    method: "invisible-watermark-dwtdctsvd"
  },
  {
    id: "alg-hidden",
    name: "HiDDeN",
    version: "packaged",
    status: "enabled",
    requiresGpu: true
  },
  {
    id: "alg-ssl-watermarking",
    name: "SSL Watermarking",
    version: "packaged",
    status: "enabled",
    requiresGpu: true
  },
  {
    id: "alg-stegastamp",
    name: "StegaStamp",
    version: "packaged",
    status: "enabled",
    requiresGpu: true
  }
];

export const attacks: AttackPreset[] = [
  { id: "atk-identity", name: "Identity", method: "identity", strengths: [0] },
  { id: "atk-jpeg", name: "JPEG Compression", method: "jpeg", strengths: [0.25, 0.5, 0.75], recommended: true },
  { id: "atk-gaussian-blur", name: "Gaussian Blur", method: "gaussian_blur", strengths: [0.2, 0.4, 0.6] },
  { id: "atk-resized-crop", name: "Resized Crop", method: "resized_crop", strengths: [0.1, 0.3, 0.5] }
];

export const artifacts: ModelArtifact[] = [
  { id: "wgt-hidden", name: "combined-noise--epoch-400.pyt", checksum: "local", size: "HiDDeN" },
  { id: "wgt-ssl", name: "dino_r50_plus.pth", checksum: "local", size: "SSL" },
  { id: "wgt-stegastamp", name: "encoder/decoder checkpoint", checksum: "local", size: "StegaStamp" }
];

export const recentRuns: Array<{
  id: string;
  name: string;
  status: RunStatus;
  cells: number;
  updatedAt: string;
}> = [
  { id: "run_20260626_001", name: "JPEG baseline sweep", status: "running", cells: 18, updatedAt: "19:45" },
  { id: "run_20260625_004", name: "Demo smoke set", status: "succeeded", cells: 3, updatedAt: "Yesterday" },
  { id: "run_20260624_002", name: "Uploaded package sandbox", status: "failed", cells: 6, updatedAt: "Jun 24" }
];
