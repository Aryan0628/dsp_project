"use client";

import { useState } from "react";
import type { AnalysisResult } from "@/lib/types";

const STAGES = [
  {
    key: "original" as const,
    stage: "1",
    title: "Input",
    blurb: "The uploaded slice, as acquired.",
  },
  {
    key: "preprocessed" as const,
    stage: "2",
    title: "Preprocessed",
    blurb: "Grayscale, resized to 256×256, median + bilateral denoising, normalised.",
  },
  {
    key: "enhanced" as const,
    stage: "3",
    title: "Enhanced",
    blurb: "CLAHE for local contrast, then unsharp masking for edge definition.",
  },
  {
    key: "brain_mask" as const,
    stage: "4a",
    title: "Skull stripped",
    blurb: "The brain region isolated, so skull brightness cannot dominate the statistics.",
  },
  {
    key: "segmentation_mask" as const,
    stage: "4b",
    title: "Candidate mask",
    blurb: "Tissue-relative threshold plus contralateral asymmetry, then morphology.",
  },
  {
    key: "overlay" as const,
    stage: "4c",
    title: "Detected region",
    blurb: "The selected region outlined on the enhanced scan.",
  },
];

/**
 * Every intermediate stage, in pipeline order.
 *
 * This gallery is the project's actual argument, not decoration. The claim that a hybrid
 * CV+ML system is auditable in a way a raw CNN is not only holds if the intermediate
 * stages are visible - a clinician can see WHICH region was measured and reject the call
 * when the segmenter has latched onto a skull artefact. A verdict with no visible working
 * is the black box we said we were avoiding.
 */
export function StageGallery({ result }: { result: AnalysisResult }) {
  const [zoomed, setZoomed] = useState<string | null>(null);
  const images = result.images;
  if (!images) return null;

  return (
    <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold">Pipeline stages</h2>
        <span className="text-[11px] text-[var(--text-dim)]">click to enlarge</span>
      </div>
      <p className="mb-4 text-[11px] text-[var(--text-dim)]">
        Every step the system took to reach its conclusion, in order.
      </p>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {STAGES.map((s) => (
          <figure key={s.key} className="group">
            <button
              onClick={() => setZoomed(images[s.key])}
              className="block w-full overflow-hidden rounded-lg border border-[var(--border)] bg-black transition-colors hover:border-[var(--accent-dim)]"
              aria-label={`Enlarge: ${s.title}`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={images[s.key]}
                alt={`Stage ${s.stage}: ${s.title}`}
                className="pixelated aspect-square w-full object-contain"
              />
            </button>
            <figcaption className="mt-1.5">
              <p className="text-xs font-medium">
                <span className="tabular mr-1.5 text-[var(--text-dim)]">{s.stage}</span>
                {s.title}
              </p>
              <p className="mt-0.5 text-[10px] leading-snug text-[var(--text-dim)]">
                {s.blurb}
              </p>
            </figcaption>
          </figure>
        ))}
      </div>

      {zoomed && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Enlarged scan"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-6"
          onClick={() => setZoomed(null)}
          onKeyDown={(e) => e.key === "Escape" && setZoomed(null)}
          tabIndex={-1}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={zoomed}
            alt="Enlarged MRI stage"
            className="pixelated max-h-full max-w-full rounded-lg object-contain"
          />
          <button
            onClick={() => setZoomed(null)}
            className="absolute right-6 top-6 rounded-md border border-[var(--border-strong)] bg-[var(--bg-panel)] px-3 py-1.5 text-xs"
          >
            Close
          </button>
        </div>
      )}
    </section>
  );
}
