"use client";

import type { AnalysisResult } from "@/lib/types";

const DESCRIPTIONS: Record<string, string> = {
  "No Tumor": "No suspicious mass on this slice.",
  Glioma: "From glial cells inside the brain tissue (intra-axial); typically deep in white matter.",
  Meningioma: "From the meninges lining the skull (extra-axial); typically against the brain's edge.",
  Pituitary: "From the pituitary gland at the skull base; typically central, on the midline.",
};

/**
 * Full per-class probability distribution.
 *
 * Showing the whole vector rather than the winning label is the point. On the typing task
 * the model's second choice carries real information: a 52/41 split between meningioma and
 * glioma is a case for a human to look at closely, and it is indistinguishable from a
 * 95/2 case if the interface only ever prints the argmax. Hiding the runner-up would make
 * the confident and the borderline look identical.
 */
export function ProbabilityBars({ result }: { result: AnalysisResult }) {
  const { prediction: p, model } = result;
  if (!p.probabilities || p.probabilities.length !== p.class_names.length) return null;

  const rows = p.class_names
    .map((name, i) => ({
      name,
      prob: p.probabilities![i],
      deep: model.deep_probabilities?.[i] ?? null,
    }))
    .sort((a, b) => b.prob - a.prob);

  const showDeep = model.deep_probabilities != null && model.branch !== "classical";

  return (
    <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold">Class probabilities</h2>
        {showDeep && (
          <span className="text-[11px] text-[var(--text-dim)]">
            bar = final blend · tick = CNN alone
          </span>
        )}
      </div>

      <ul className="space-y-3.5">
        {rows.map((r) => {
          const isTop = r.name === p.label_name;
          return (
            <li key={r.name}>
              <div className="mb-1 flex items-baseline justify-between gap-3">
                <span
                  className={`text-sm ${isTop ? "font-semibold text-[var(--text)]" : "text-[var(--text-muted)]"}`}
                >
                  {r.name}
                </span>
                <span
                  className={`tabular text-sm ${isTop ? "font-semibold text-[var(--accent)]" : "text-[var(--text-dim)]"}`}
                >
                  {(r.prob * 100).toFixed(1)}%
                </span>
              </div>

              <div className="relative h-2 overflow-hidden rounded-full bg-[var(--bg)]">
                <div
                  className="h-full rounded-full transition-[width] duration-500"
                  style={{
                    width: `${Math.max(r.prob * 100, 0.6)}%`,
                    background: isTop ? "var(--accent)" : "var(--border-strong)",
                  }}
                />
                {/* The CNN's own opinion, marked on the same axis. When the tick sits far
                    from the bar end, the two halves of the system disagree - which is
                    exactly the case a reviewer should be told about rather than have
                    averaged away silently. */}
                {showDeep && r.deep != null && (
                  <span
                    className="absolute top-0 h-full w-px bg-[var(--danger)]"
                    style={{ left: `${Math.min(Math.max(r.deep * 100, 0.3), 99.7)}%` }}
                    title={`CNN alone: ${(r.deep * 100).toFixed(1)}%`}
                  />
                )}
              </div>

              {DESCRIPTIONS[r.name] && (
                <p className="mt-1 text-[11px] leading-snug text-[var(--text-dim)]">
                  {DESCRIPTIONS[r.name]}
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
