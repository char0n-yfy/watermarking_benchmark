import type { AlgorithmVersion, AttackPreset, DatasetVersion, ModelArtifact, RunStatus } from "./types";

export const datasets: DatasetVersion[] = [
  { id: "ds-coco-v1", name: "MS-COCO validation slice", sampleCount: 500, version: "v1" },
  { id: "ds-diffusiondb-v1", name: "DiffusionDB curated", sampleCount: 1200, version: "v1" },
  { id: "ds-demo-v1", name: "Demo smoke set", sampleCount: 32, version: "v1" }
];

export const algorithms: AlgorithmVersion[] = [
  {
    id: "alg-dct-qim-001",
    name: "DCT-QIM Baseline",
    version: "0.1.0",
    status: "enabled",
    requiresGpu: false
  },
  {
    id: "alg-stable-signature-adapter",
    name: "Stable Signature Adapter",
    version: "uploaded",
    status: "reviewed",
    requiresGpu: true
  },
  {
    id: "alg-custom-lab-drop",
    name: "Lab Upload Package",
    version: "pending",
    status: "uploaded",
    requiresGpu: true
  }
];

export const attacks: AttackPreset[] = [
  { id: "atk-identity", name: "Identity", method: "identity", strengths: [0] },
  { id: "atk-jpeg-sweep", name: "JPEG sweep", method: "jpeg", strengths: [0.25, 0.5, 0.75] },
  { id: "atk-blur-sweep", name: "Blur sweep", method: "gaussian_blur", strengths: [0.2, 0.4, 0.6] },
  { id: "atk-crop-sweep", name: "Crop sweep", method: "resized_crop", strengths: [0.1, 0.3, 0.5] }
];

export const artifacts: ModelArtifact[] = [
  { id: "wgt-dct-default", name: "dct-qim-default.json", checksum: "sha256:6b5e...", size: "4 KB" },
  { id: "wgt-stable-sig", name: "stable-signature-decoder.pt", checksum: "sha256:fd02...", size: "91 MB" }
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
