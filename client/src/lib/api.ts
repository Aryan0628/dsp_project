/**
 * Client for the Python backend.
 *
 * Requests go from the BROWSER straight to FastAPI, not through a Next.js route handler.
 * That is deliberate: proxying would mean every uploaded scan is copied into the Node
 * process and buffered there, and the images come back as multi-megabyte base64 payloads
 * that would then be copied a second time. Direct calls keep one copy in flight and make
 * the backend the only component that ever holds patient data.
 */

import type {
  AnalysisResult,
  ApiErrorDetail,
  ClassesResult,
  HealthResult,
  ModelsResult,
  Task,
  TrainingMetrics,
} from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:3000";

/** An error carrying the backend's structured detail, so the UI can show the fix. */
export class ApiError extends Error {
  status: number;
  detail?: ApiErrorDetail;

  constructor(message: string, status: number, detail?: ApiErrorDetail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  let detail: ApiErrorDetail | undefined;
  let message = `Request failed (${res.status} ${res.statusText})`;
  try {
    const body = await res.json();
    // FastAPI wraps HTTPException payloads in `detail`. Ours are objects; validation
    // errors from FastAPI itself are arrays, hence the shape check.
    if (body?.detail && typeof body.detail === "object" && !Array.isArray(body.detail)) {
      detail = body.detail as ApiErrorDetail;
      message = detail.message ?? message;
    } else if (typeof body?.detail === "string") {
      message = body.detail;
    }
  } catch {
    /* response had no JSON body; the status-line message stands */
  }
  return new ApiError(message, res.status, detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch (cause) {
    // fetch only rejects on network-level failure, which here almost always means the
    // backend is not running. Say that, rather than surfacing "Failed to fetch".
    throw new ApiError(
      `Cannot reach the API at ${API_URL}. Is the backend running?`,
      0,
      { error: "network", message: "Backend unreachable", fix: "Run: ./run.sh serve" },
    );
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResult> {
  return request<HealthResult>("/health", { cache: "no-store" });
}

export function getModels(): Promise<ModelsResult> {
  return request<ModelsResult>("/models", { cache: "no-store" });
}

export function getClasses(): Promise<ClassesResult> {
  return request<ClassesResult>("/classes");
}

export function getMetrics(task: Task): Promise<TrainingMetrics> {
  return request<TrainingMetrics>(`/metrics?multiclass=${task === "multiclass"}`, {
    cache: "no-store",
  });
}

export function analyze(
  file: File,
  task: Task,
  opts: { includeImages?: boolean } = {},
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({
    multiclass: String(task === "multiclass"),
    include_images: String(opts.includeImages ?? true),
  });
  // No Content-Type header: the browser must set it itself so the multipart boundary is
  // included. Setting it manually is the classic way to make a file upload fail.
  return request<AnalysisResult>(`/analyze?${params}`, { method: "POST", body: form });
}
