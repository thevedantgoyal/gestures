import { type SignGesture } from "./signVocab";

export type SignTrainStatus = "collecting" | "trainable" | "ready";
export type SignLivePath = "rules" | "hybrid" | "model";
export type SignTrainSource = "heuristic" | "hybrid" | "model";

export type SignClassMetrics = {
  precision: number;
  recall: number;
  f1: number;
  support: number;
};

export type SignTrainLast = {
  version: string;
  trained_at?: string;
  train_accuracy?: number;
  val_accuracy?: number;
  accuracy_split?: string;
  classes?: string[];
  per_class?: Record<string, SignClassMetrics>;
  warnings?: string[];
  split?: string;
  n_train_rows?: number;
  n_val_rows?: number;
  hyperparameters?: Record<string, number | string>;
};

export type SignLiveTest = {
  recognizer: string | null;
  label: string | null;
  prob: number | null;
  agree: boolean | null;
  available: boolean;
  version: string | null;
};

export type SignTrainGestureStats = {
  gesture: SignGesture;
  meaning: string;
  type: "pose" | "motion";
  gold_holds: number;
  status: SignTrainStatus;
  progress: number;
  live_path: SignLivePath;
  metrics: SignClassMetrics | null;
};

export type SignTrainStats = {
  source: SignTrainSource;
  model: {
    available: boolean;
    version: string | null;
  };
  feature_version: number;
  trainable_threshold: number;
  promote_threshold: number;
  min_train_holds?: number;
  static_sign_count: number;
  overall: {
    gold_holds: number;
    trainable_signs: number;
    promote_ready_signs: number;
    progress: number;
  };
  last_train?: SignTrainLast | null;
  can_train?: boolean;
  can_promote?: boolean;
  train_blockers?: string[];
  train_warnings?: string[];
  gestures: SignTrainGestureStats[];
};

export function trainStatusLabel(status: SignTrainStatus): string {
  switch (status) {
    case "collecting":
      return "Collecting";
    case "trainable":
      return "Trainable";
    case "ready":
      return "Ready to promote";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

export function livePathLabel(path: SignLivePath): string {
  switch (path) {
    case "rules":
      return "Rules";
    case "hybrid":
      return "Hybrid";
    case "model":
      return "Model";
    default: {
      const exhaustive: never = path;
      return exhaustive;
    }
  }
}

export function sourceToLivePath(source: SignTrainSource): SignLivePath {
  switch (source) {
    case "heuristic":
      return "rules";
    case "hybrid":
      return "hybrid";
    case "model":
      return "model";
    default: {
      const exhaustive: never = source;
      return exhaustive;
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

export function isSignTrainStats(value: unknown): value is SignTrainStats {
  if (!isRecord(value)) {
    return false;
  }
  return Array.isArray(value.gestures) && isRecord(value.overall);
}

export function liveTestFromDebug(
  debug: Record<string, unknown> | null,
): SignLiveTest | null {
  if (!debug) {
    return null;
  }
  const model = debug.model;
  if (!isRecord(model)) {
    return null;
  }
  return {
    recognizer:
      typeof debug.recognizer === "string" ? debug.recognizer : null,
    label: typeof model.label === "string" ? model.label : null,
    prob: typeof model.prob === "number" ? model.prob : null,
    agree: typeof model.agree === "boolean" ? model.agree : null,
    available: model.available === true,
    version: typeof model.version === "string" ? model.version : null,
  };
}

export function percentLabel(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}
