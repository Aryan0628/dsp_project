/**
 * Types mirroring the FastAPI response shapes in `src/braintumor/api.py`.
 *
 * These are hand-written rather than generated. The API is small and stable, and a
 * generator would add a build step and a schema-drift failure mode for four screens'
 * worth of fields. If the API grows, revisit that trade.
 */

export type Task = "binary" | "multiclass";

export interface Prediction {
  label: number;
  label_name: string;
  is_tumor: boolean;
  confidence: number;
  tumor_probability: number;
  class_names: string[];
  /** Per-class probabilities, aligned with `class_names`. Null if the model has none. */
  probabilities: number[] | null;
  threshold: number;
}

export interface ModelInfo {
  name: string;
  /** Which half of the system produced the answer: classical features, CNN, or a blend. */
  branch: "classical" | "deep" | "ensemble";
  blend_weight_deep: number;
  deep_probabilities: number[] | null;
}

export interface RegionProperties {
  centroid: [number, number];
  bbox: [number, number, number, number];
  area: number;
  area_ratio: number;
  perimeter: number;
  circularity: number;
  solidity: number;
  extent: number;
  aspect_ratio: number;
  mean_intensity: number;
  max_intensity: number;
  brightness_ratio: number;
}

export interface SegmentationInfo {
  tumor_found: boolean;
  n_candidates: number;
  threshold: number;
  properties: RegionProperties | null;
}

export interface EnhancementInfo {
  contrast_before: number;
  contrast_after: number;
  entropy_before: number;
  entropy_after: number;
  sharpness_before: number;
  sharpness_after: number;
}

/** Every pipeline stage, as base64 data URIs ready for an <img src>. */
export interface StageImages {
  original: string;
  preprocessed: string;
  enhanced: string;
  brain_mask: string;
  segmentation_mask: string;
  overlay: string;
}

export interface LocationInfo {
  loc_found?: number;
  loc_radial?: number;
  loc_midline_abs?: number;
  loc_depth_centroid?: number;
  loc_border_contact?: number;
  loc_band_sup?: number;
  loc_band_mid?: number;
  loc_band_inf?: number;
}

export interface AnalysisResult {
  prediction: Prediction;
  model: ModelInfo;
  segmentation: SegmentationInfo;
  enhancement: EnhancementInfo;
  images?: StageImages;
  location?: LocationInfo;
  meta: { filename: string; elapsed_ms: number; task: Task };
  disclaimer: string;
}

export interface HealthResult {
  status: string;
  version: string;
  models: Record<string, { trained: boolean; path: string }>;
  torch_available: boolean;
  device: string | null;
}

export interface ModelsResult {
  [task: string]: {
    trained: boolean;
    model_name?: string;
    task?: string;
    dataset?: string;
    source?: string;
    class_names?: string[];
    n_features?: number;
    threshold?: number;
    has_deep_branch?: boolean;
    blend_weight_deep?: number;
    shipped_branch?: string;
    deep_load_error?: string | null;
  };
}

export interface ClassesResult {
  binary: string[];
  multiclass: string[];
  descriptions: Record<string, string>;
}

export interface TrainingMetrics {
  task: string;
  dataset_source: string;
  split_kind: string;
  n_images: number;
  n_train: number;
  n_test: number;
  n_features: number;
  best_model: string;
  class_counts: Record<string, number>;
  test_metrics: {
    accuracy: number;
    balanced_accuracy: number;
    macro_f1?: number;
    roc_auc?: number;
    roc_auc_ovr?: number;
    sensitivity_recall?: number;
    specificity?: number;
    precision?: number;
    f1?: number;
    per_class_recall?: Record<string, number>;
    confusion_matrix_full?: number[][];
    class_names?: string[];
    confusion_matrix?: {
      true_negative: number;
      false_positive: number;
      false_negative: number;
      true_positive: number;
    };
  };
  test_metrics_tuned?: TrainingMetrics["test_metrics"] | null;
  threshold_info?: { threshold: number; objective?: string; note?: string } | null;
  deep_info?: {
    arch: string;
    epochs: number;
    device: string;
    shipped_branch: string;
    train_seconds: number;
    blend: { weight_deep: number; val_balanced_accuracy: number };
    test_metrics_classical_only?: { accuracy: number; balanced_accuracy: number };
    test_metrics_deep_only?: { accuracy: number; balanced_accuracy: number };
    test_metrics_ensemble?: { accuracy: number; balanced_accuracy: number };
  } | null;
  train_seconds: number;
}

/** Shape of the structured `detail` object FastAPI returns on our error paths. */
export interface ApiErrorDetail {
  error: string;
  message: string;
  fix?: string;
  accepted?: string[];
}
