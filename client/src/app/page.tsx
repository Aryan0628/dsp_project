"use client";

import { useCallback, useState } from "react";
import { analyze, ApiError } from "@/lib/api";
import type { AnalysisResult, Task } from "@/lib/types";
import { UploadZone } from "@/components/UploadZone";
import { VerdictCard } from "@/components/VerdictCard";
import { ProbabilityBars } from "@/components/ProbabilityBars";
import { StageGallery } from "@/components/StageGallery";
import { MeasurementPanel } from "@/components/MeasurementPanel";
import { TaskToggle } from "@/components/TaskToggle";
import { ErrorPanel } from "@/components/ErrorPanel";

export default function AnalysePage() {
  const [task, setTask] = useState<Task>("multiclass");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | Error | null>(null);

  const run = useCallback(
    async (f: File, t: Task) => {
      setBusy(true);
      setError(null);
      // Clear the previous result before the request rather than after it. Leaving the old
      // verdict on screen while a new scan is being analysed invites reading it as the
      // answer for the new file, which in this domain is the worst possible confusion.
      setResult(null);
      try {
        setResult(await analyze(f, t));
      } catch (e) {
        setError(e instanceof Error ? e : new Error(String(e)));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const onFile = useCallback(
    (f: File) => {
      setFile(f);
      void run(f, task);
    },
    [run, task],
  );

  const onTaskChange = useCallback(
    (t: Task) => {
      setTask(t);
      // Re-run automatically when a scan is already loaded: switching task is a question
      // about the SAME image, so making the user re-upload it would be busywork.
      if (file) void run(file, t);
    },
    [file, run],
  );

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="mb-8 max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Analyse a brain MRI slice</h1>
        <p className="mt-2 text-sm leading-relaxed text-[var(--text-muted)]">
          The scan is denoised and contrast-enhanced with OpenCV, the suspicious region is
          segmented and measured, and a classifier reaches a verdict from those
          measurements. Every intermediate stage is shown, so the conclusion can be checked
          rather than taken on trust.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        {/* ---------------------------------------------------------------- controls */}
        <div className="space-y-5 lg:sticky lg:top-24 lg:self-start">
          <TaskToggle value={task} onChange={onTaskChange} disabled={busy} />
          <UploadZone onFile={onFile} disabled={busy} filename={file?.name ?? null} />

          {file && !busy && (
            <button
              onClick={() => void run(file, task)}
              className="w-full rounded-lg border border-[var(--border-strong)] bg-[var(--bg-panel)] px-4 py-2.5 text-sm font-medium transition-colors hover:border-[var(--accent-dim)] hover:text-[var(--accent)]"
            >
              Re-run analysis
            </button>
          )}

          {busy && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-panel)] px-4 py-3 text-sm text-[var(--text-muted)]">
              <span className="animate-pulse-subtle">
                Running stages 2–6 on {file?.name ?? "the scan"}…
              </span>
            </div>
          )}

          {error && <ErrorPanel error={error} />}

          {!result && !busy && !error && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-panel)] p-4 text-xs leading-relaxed text-[var(--text-dim)]">
              <p className="mb-2 font-medium text-[var(--text-muted)]">No scan loaded</p>
              <p>
                Upload a single 2D brain MRI slice. Sample images live in{" "}
                <code className="rounded bg-[var(--bg)] px-1 py-0.5">data/kaggle4/test/</code>{" "}
                inside the project.
              </p>
            </div>
          )}
        </div>

        {/* ---------------------------------------------------------------- results */}
        <div className="space-y-5">
          {result ? (
            <>
              <VerdictCard result={result} />
              {task === "multiclass" && <ProbabilityBars result={result} />}
              <StageGallery result={result} />
              <MeasurementPanel result={result} />
            </>
          ) : (
            <EmptyState busy={busy} />
          )}
        </div>
      </div>
    </div>
  );
}

function EmptyState({ busy }: { busy: boolean }) {
  return (
    <div className="flex min-h-[420px] items-center justify-center rounded-xl border border-dashed border-[var(--border)] bg-[var(--bg-panel)]/40">
      <div className="max-w-sm px-6 text-center">
        <svg
          className={`mx-auto mb-4 ${busy ? "animate-pulse-subtle" : ""}`}
          width="44"
          height="44"
          viewBox="0 0 32 32"
          fill="none"
          aria-hidden="true"
        >
          <circle cx="16" cy="16" r="13" stroke="var(--border-strong)" strokeWidth="1.5" />
          <path
            d="M16 4.5v23M8 9c3 2.2 3 5.4 0 7.6M24 9c-3 2.2-3 5.4 0 7.6"
            stroke="var(--border-strong)"
            strokeWidth="1.2"
            strokeLinecap="round"
          />
        </svg>
        <p className="text-sm text-[var(--text-muted)]">
          {busy ? "Analysing…" : "Results will appear here"}
        </p>
        <p className="mt-1.5 text-xs text-[var(--text-dim)]">
          {busy
            ? "Preprocessing, enhancement, segmentation, feature extraction, classification."
            : "Upload a slice to begin."}
        </p>
      </div>
    </div>
  );
}
