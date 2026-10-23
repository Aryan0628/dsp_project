"use client";

import { ApiError } from "@/lib/api";

/**
 * Error display that shows the fix, not just the failure.
 *
 * The backend returns a structured `detail` with an `error` code and, where one exists, a
 * `fix` string containing the exact command to run. Surfacing that turns the two most
 * common failures - backend not started, model not trained - from dead ends into a line
 * the user can copy.
 */
export function ErrorPanel({ error }: { error: Error }) {
  const api = error instanceof ApiError ? error : null;
  const detail = api?.detail;

  return (
    <div
      role="alert"
      className="rounded-lg border p-4"
      style={{
        background: "var(--error-bg)",
        borderColor: "color-mix(in srgb, var(--error) 35%, transparent)",
      }}
    >
      <p className="text-sm font-medium" style={{ color: "var(--error)" }}>
        {detail?.error === "model_not_trained"
          ? "No trained model"
          : detail?.error === "network"
            ? "Backend unreachable"
            : "Analysis failed"}
      </p>

      <p className="mt-1.5 text-xs leading-relaxed text-[var(--text-muted)]">
        {error.message}
      </p>

      {detail?.fix && (
        <div className="mt-3">
          <p className="mb-1 text-[10px] uppercase tracking-wider text-[var(--text-dim)]">
            To fix
          </p>
          <code className="block overflow-x-auto rounded bg-[var(--bg)] px-2.5 py-1.5 text-[11px] text-[var(--accent)]">
            {detail.fix}
          </code>
        </div>
      )}

      {detail?.accepted && (
        <p className="mt-2 text-[11px] text-[var(--text-dim)]">
          Accepted types: {detail.accepted.join(", ")}
        </p>
      )}

      {api && api.status > 0 && (
        <p className="mt-2 text-[10px] text-[var(--text-dim)]">HTTP {api.status}</p>
      )}
    </div>
  );
}
