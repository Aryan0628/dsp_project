"use client";

import { useEffect, useState } from "react";
import { getMetrics, getModels } from "@/lib/api";
import type { ModelsResult, Task, TrainingMetrics } from "@/lib/types";
import { TaskToggle } from "@/components/TaskToggle";
import { ErrorPanel } from "@/components/ErrorPanel";
import { ConfusionMatrix } from "@/components/ConfusionMatrix";

/**
 * Measured performance of the shipped models, read from the last training report.
 *
 * These are held-out test numbers on the dataset's OWN train/test split, not a random
 * split of pooled data. That distinction costs several apparent accuracy points and is the
 * reason the numbers here are lower than most published projects on the same dataset - a
 * random split puts near-identical slices of one patient on both sides and grades the model
 * on images it has effectively already seen.
 */
/** What a completed fetch produced, tagged with the task it was fetched for. */
interface Loaded {
  task: Task;
  metrics?: TrainingMetrics;
  models?: ModelsResult | null;
  error?: Error;
}

export default function PerformancePage() {
  const [task, setTask] = useState<Task>("multiclass");
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  // `busy` is DERIVED from whether the loaded data matches the task currently selected,
  // rather than being a separate flag flipped at the top of the effect. Setting state
  // synchronously in an effect body triggers a second render pass before paint, and it
  // also makes "loading" and "loaded" two sources of truth that can disagree - here they
  // cannot, because there is only one.
  const current = loaded?.task === task ? loaded : null;
  const busy = current === null;
  const metrics = current?.metrics ?? null;
  const models = current?.models ?? null;
  const error = current?.error ?? null;

  useEffect(() => {
    let cancelled = false;

    Promise.all([getMetrics(task), getModels().catch(() => null)])
      .then(([m, mods]) => {
        if (!cancelled) setLoaded({ task, metrics: m, models: mods });
      })
      .catch((e) => {
        if (!cancelled) {
          setLoaded({ task, error: e instanceof Error ? e : new Error(String(e)) });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [task]);

  const tm = metrics?.test_metrics;
  // For the binary task the SHIPPED numbers are the tuned-threshold ones, so those are
  // what this page must show. Reporting the default-0.5 figures would describe a system
  // nobody is running.
  const shipped = metrics?.test_metrics_tuned ?? tm;

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="mb-8 max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Measured performance</h1>
        <p className="mt-2 text-sm leading-relaxed text-[var(--text-muted)]">
          Held-out test results from the most recent training run. The split is the
          dataset&apos;s own, so slices from one patient never appear on both sides — which
          costs several apparent accuracy points and is the reason to trust what is left.
        </p>
      </div>

      <div className="mb-6 max-w-md">
        <TaskToggle value={task} onChange={setTask} disabled={busy} />
      </div>

      {error && <ErrorPanel error={error} />}

      {busy && !metrics && (
        <p className="animate-pulse-subtle text-sm text-[var(--text-muted)]">
          Loading the training report…
        </p>
      )}

      {metrics && shipped && (
        <div className="space-y-6">
          {/* ---- headline metrics */}
          <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            <Stat label="Accuracy" value={pct(shipped.accuracy)} primary />
            <Stat label="Balanced accuracy" value={pct(shipped.balanced_accuracy)} primary />
            {task === "binary" ? (
              <>
                <Stat label="Sensitivity" value={pct(shipped.sensitivity_recall)} primary />
                <Stat label="Specificity" value={pct(shipped.specificity)} />
                <Stat
                  label="ROC-AUC"
                  value={shipped.roc_auc != null ? shipped.roc_auc.toFixed(4) : "—"}
                />
              </>
            ) : (
              <>
                <Stat label="Macro F1" value={pct(shipped.macro_f1)} />
                <Stat
                  label="ROC-AUC (ovr)"
                  value={shipped.roc_auc_ovr != null ? shipped.roc_auc_ovr.toFixed(4) : "—"}
                />
                <Stat label="Test images" value={String(metrics.n_test)} />
              </>
            )}
          </section>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* ---- confusion matrix */}
            <ConfusionMatrix metrics={metrics} shipped={shipped} task={task} />

            <div className="space-y-6">
              {/* ---- per-class recall */}
              {shipped.per_class_recall && (
                <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
                  <h2 className="mb-1 text-sm font-semibold">Per-class recall</h2>
                  <p className="mb-4 text-[11px] text-[var(--text-dim)]">
                    Of all scans truly in each class, the share the model found.
                  </p>
                  <ul className="space-y-3">
                    {Object.entries(shipped.per_class_recall).map(([name, v]) => (
                      <li key={name}>
                        <div className="mb-1 flex justify-between text-xs">
                          <span className="text-[var(--text-muted)]">{name}</span>
                          <span className="tabular text-[var(--text)]">{pct(v)}</span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-[var(--bg)]">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${v * 100}%`,
                              // A recall this low is a finding, not a decoration. Colouring
                              // it differently stops a weak class from being skimmed past.
                              background: v < 0.5 ? "var(--danger)" : "var(--accent)",
                            }}
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {/* ---- run metadata */}
              <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
                <h2 className="mb-4 text-sm font-semibold">This run</h2>
                <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
                  <Meta label="Dataset" value={metrics.dataset_source} wide />
                  <Meta label="Split" value={`${metrics.split_kind} · ${metrics.n_train} train / ${metrics.n_test} test`} wide />
                  <Meta label="Classical model" value={metrics.best_model} />
                  <Meta label="Features" value={String(metrics.n_features)} />
                  <Meta label="Training time" value={`${metrics.train_seconds.toFixed(0)} s`} />
                  {metrics.threshold_info && (
                    <Meta
                      label="Decision threshold"
                      value={metrics.threshold_info.threshold.toFixed(3)}
                    />
                  )}
                  {models?.[task]?.shipped_branch && (
                    <Meta label="Shipped branch" value={models[task].shipped_branch!} />
                  )}
                </dl>
              </section>
            </div>
          </div>

          {/* ---- deep branch comparison */}
          {metrics.deep_info && <DeepBranchPanel info={metrics.deep_info} />}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- deep panel */
function DeepBranchPanel({
  info,
}: {
  info: NonNullable<TrainingMetrics["deep_info"]>;
}) {
  const rows = [
    { label: "Engineered features only", m: info.test_metrics_classical_only, key: "classical" },
    { label: "Fine-tuned CNN only", m: info.test_metrics_deep_only, key: "deep" },
    { label: "Ensemble of both", m: info.test_metrics_ensemble, key: "ensemble" },
  ].filter((r) => r.m);

  return (
    <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
      <h2 className="mb-1 text-sm font-semibold">
        What the CNN branch contributed
      </h2>
      <p className="mb-4 max-w-3xl text-[11px] leading-relaxed text-[var(--text-dim)]">
        A {info.arch} pretrained on ImageNet, fine-tuned for {info.epochs} epochs on{" "}
        {info.device} ({info.train_seconds.toFixed(0)} s). The blend weight was chosen on a
        validation split carved out of the training set — the test set below played no part
        in selecting it, which is why all three rows can be shown side by side without the
        comparison becoming circular.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] text-sm">
          <thead>
            <tr className="border-b border-[var(--border)] text-left text-[11px] uppercase tracking-wider text-[var(--text-dim)]">
              <th className="pb-2 font-normal">Branch</th>
              <th className="pb-2 text-right font-normal">Accuracy</th>
              <th className="pb-2 text-right font-normal">Balanced</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isShipped = r.key === info.shipped_branch;
              return (
                <tr key={r.key} className="border-b border-[var(--border)] last:border-0">
                  <td className="py-2.5">
                    {r.label}
                    {isShipped && (
                      <span className="ml-2 rounded bg-[var(--accent)]/15 px-1.5 py-0.5 text-[10px] text-[var(--accent)]">
                        shipped
                      </span>
                    )}
                  </td>
                  <td className="tabular py-2.5 text-right">{pct(r.m!.accuracy)}</td>
                  <td
                    className="tabular py-2.5 text-right"
                    style={{ color: isShipped ? "var(--accent)" : undefined }}
                  >
                    {pct(r.m!.balanced_accuracy)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-[11px] text-[var(--text-dim)]">
        CNN weight in the blend: {info.blend.weight_deep.toFixed(2)} · chosen at{" "}
        {pct(info.blend.val_balanced_accuracy)} validation balanced accuracy.
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------------------- bits */
function pct(v: number | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(2)}%`;
}

function Stat({
  label,
  value,
  primary,
}: {
  label: string;
  value: string;
  primary?: boolean;
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-4">
      <p className="text-[11px] text-[var(--text-dim)]">{label}</p>
      <p
        className="tabular mt-1 text-2xl font-semibold"
        style={{ color: primary ? "var(--accent)" : "var(--text)" }}
      >
        {value}
      </p>
    </div>
  );
}

function Meta({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={wide ? "col-span-2" : ""}>
      <dt className="text-[11px] text-[var(--text-dim)]">{label}</dt>
      <dd className="tabular mt-0.5 text-xs text-[var(--text)]">{value}</dd>
    </div>
  );
}
