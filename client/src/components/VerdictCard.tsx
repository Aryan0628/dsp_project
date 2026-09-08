"use client";

import type { AnalysisResult } from "@/lib/types";

/**
 * The headline result: what the system concluded, how sure it is, and - importantly -
 * which part of the system concluded it.
 *
 * The confidence figure needs a note. For the binary task the backend rescales it around
 * the TUNED decision threshold rather than 0.5, because the shipped threshold is well
 * below 0.5. Without that rescaling a probability of 0.23 against a 0.15 threshold would
 * display as "23% confident" while the model was in fact making a comfortable positive
 * call - the number would read as the opposite of what the system means.
 */
export function VerdictCard({ result }: { result: AnalysisResult }) {
  const { prediction: p, model, meta } = result;
  const positive = p.is_tumor;

  const tone = positive
    ? { fg: "var(--danger)", bg: "var(--danger-bg)", border: "var(--danger)" }
    : { fg: "var(--safe)", bg: "var(--safe-bg)", border: "var(--safe)" };

  return (
    <section
      className="rounded-xl border p-5"
      style={{ background: tone.bg, borderColor: `color-mix(in srgb, ${tone.border} 35%, transparent)` }}
      aria-live="polite"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-wider text-[var(--text-dim)]">
            {meta.task === "multiclass" ? "Tumor type" : "Screening result"}
          </p>
          <p className="mt-1 text-3xl font-semibold tracking-tight" style={{ color: tone.fg }}>
            {p.label_name}
          </p>
          <p className="mt-1.5 text-xs text-[var(--text-muted)]">
            {positive
              ? "Flagged for radiologist review."
              : "No suspicious mass detected on this slice."}
          </p>
        </div>

        <div className="text-right">
          <p className="text-[11px] uppercase tracking-wider text-[var(--text-dim)]">
            Confidence
          </p>
          <p className="tabular text-3xl font-semibold" style={{ color: tone.fg }}>
            {(p.confidence * 100).toFixed(1)}%
          </p>
          {meta.task === "binary" && (
            <p className="tabular mt-1 text-[11px] text-[var(--text-dim)]">
              P(tumor) {(p.tumor_probability * 100).toFixed(1)}% · threshold{" "}
              {p.threshold.toFixed(3)}
            </p>
          )}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 border-t border-[var(--border)] pt-3 text-[11px]">
        <Chip label="Model" value={model.name} />
        <Chip
          label="Decided by"
          value={
            model.branch === "ensemble"
              ? `ensemble · CNN weight ${model.blend_weight_deep.toFixed(2)}`
              : model.branch === "deep"
                ? "fine-tuned CNN"
                : "engineered features"
          }
        />
        <Chip label="Elapsed" value={`${meta.elapsed_ms.toFixed(0)} ms`} />
      </div>

      {meta.task === "binary" && Math.abs(p.threshold - 0.5) > 1e-6 && (
        <p className="mt-3 text-[11px] leading-relaxed text-[var(--text-dim)]">
          The decision threshold is {p.threshold.toFixed(3)}, not the default 0.5. It was
          tuned on out-of-fold training predictions to maximise sensitivity within a
          specificity budget, because a missed tumor costs far more than a second review.
        </p>
      )}
    </section>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 py-1">
      <span className="text-[var(--text-dim)]">{label}</span>
      <span className="tabular text-[var(--text-muted)]">{value}</span>
    </span>
  );
}
