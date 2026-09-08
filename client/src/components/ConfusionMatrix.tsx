"use client";

import type { Task, TrainingMetrics } from "@/lib/types";

/**
 * Confusion matrix for either task.
 *
 * This gets a panel of its own because on an imbalanced clinical dataset the headline
 * accuracy hides the error that matters. A model can post a strong accuracy while missing
 * a third of the tumors, and only the matrix shows it. The binary view therefore calls out
 * false negatives explicitly rather than leaving them as one cell among four.
 */
export function ConfusionMatrix({
  metrics,
  shipped,
  task,
}: {
  metrics: TrainingMetrics;
  shipped: TrainingMetrics["test_metrics"];
  task: Task;
}) {
  if (task === "binary" && shipped.confusion_matrix) {
    const cm = shipped.confusion_matrix;
    return (
      <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
        <h2 className="mb-1 text-sm font-semibold">Confusion matrix</h2>
        <p className="mb-4 text-[11px] text-[var(--text-dim)]">
          At the shipped decision threshold, on {metrics.n_test} held-out scans.
        </p>

        <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center text-sm">
          <div />
          <Head>predicted normal</Head>
          <Head>predicted tumor</Head>

          <Side>actual normal</Side>
          <Cell value={cm.true_negative} tone="good" caption="true negative" />
          <Cell value={cm.false_positive} tone="warn" caption="false alarm" />

          <Side>actual tumor</Side>
          <Cell value={cm.false_negative} tone="bad" caption="missed tumor" />
          <Cell value={cm.true_positive} tone="good" caption="true positive" />
        </div>

        <p className="mt-4 text-[11px] leading-relaxed text-[var(--text-dim)]">
          <strong className="text-[var(--danger)]">{cm.false_negative} missed tumor
          {cm.false_negative === 1 ? "" : "s"}</strong> and {cm.false_positive} false alarm
          {cm.false_positive === 1 ? "" : "s"}. These two errors are not symmetric: a missed
          tumor sends a patient home untreated, a false alarm costs one extra review. The
          threshold is tuned accordingly.
        </p>
      </section>
    );
  }

  const full = shipped.confusion_matrix_full;
  const names = shipped.class_names;
  if (!full || !names) return null;

  const max = Math.max(...full.flat());

  return (
    <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
      <h2 className="mb-1 text-sm font-semibold">Confusion matrix</h2>
      <p className="mb-4 text-[11px] text-[var(--text-dim)]">
        Rows are the true class, columns the prediction. The diagonal is correct.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[380px] border-collapse text-sm">
          <thead>
            <tr>
              <th className="p-1.5" />
              {names.map((n) => (
                <th
                  key={n}
                  className="p-1.5 text-[10px] font-normal text-[var(--text-dim)]"
                  title={`predicted ${n}`}
                >
                  {n}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {full.map((row, i) => (
              <tr key={names[i]}>
                <th className="p-1.5 text-right text-[10px] font-normal text-[var(--text-dim)] whitespace-nowrap">
                  {names[i]}
                </th>
                {row.map((v, j) => {
                  const diag = i === j;
                  // Intensity encodes count; hue encodes right-vs-wrong. Off-diagonal mass
                  // is what a reader should be drawn to, so it gets the alerting colour.
                  const alpha = max ? 0.12 + (v / max) * 0.65 : 0.12;
                  return (
                    <td
                      key={j}
                      className="tabular p-1.5 text-center"
                      style={{
                        background:
                          v === 0
                            ? "var(--bg)"
                            : diag
                              ? `color-mix(in srgb, var(--accent) ${alpha * 100}%, var(--bg))`
                              : `color-mix(in srgb, var(--danger) ${alpha * 100}%, var(--bg))`,
                        color: v === 0 ? "var(--text-dim)" : "var(--text)",
                      }}
                      title={`${v} ${names[i]} predicted as ${names[j]}`}
                    >
                      {v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Head({ children }: { children: React.ReactNode }) {
  return (
    <div className="pb-1 text-[10px] text-[var(--text-dim)]">{children}</div>
  );
}

function Side({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center pr-2 text-right text-[10px] text-[var(--text-dim)]">
      {children}
    </div>
  );
}

function Cell({
  value,
  tone,
  caption,
}: {
  value: number;
  tone: "good" | "warn" | "bad";
  caption: string;
}) {
  const colour =
    tone === "good" ? "var(--safe)" : tone === "bad" ? "var(--danger)" : "var(--text-muted)";
  return (
    <div
      className="rounded-lg border p-3"
      style={{
        borderColor: `color-mix(in srgb, ${colour} 28%, transparent)`,
        background: `color-mix(in srgb, ${colour} 8%, var(--bg))`,
      }}
    >
      <div className="tabular text-2xl font-semibold" style={{ color: colour }}>
        {value}
      </div>
      <div className="mt-0.5 text-[10px] text-[var(--text-dim)]">{caption}</div>
    </div>
  );
}
