"use client";

import type { Task } from "@/lib/types";

const OPTIONS: { value: Task; label: string; blurb: string }[] = [
  {
    value: "binary",
    label: "Detection",
    blurb: "Tumor present or absent. The stronger model — 96.9% sensitivity.",
  },
  {
    value: "multiclass",
    label: "Typing",
    blurb: "Which kind: glioma, meningioma, pituitary, or none.",
  },
];

/**
 * Switch between the two trained models.
 *
 * The blurbs state each task's real standing rather than presenting them as equals.
 * Detection is a solved problem here; typing is materially harder and the interface should
 * not imply otherwise just because both are one click away.
 */
export function TaskToggle({
  value,
  onChange,
  disabled,
}: {
  value: Task;
  onChange: (t: Task) => void;
  disabled?: boolean;
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-1.5">
      <div
        className="grid grid-cols-2 gap-1.5"
        role="radiogroup"
        aria-label="Analysis task"
      >
        {OPTIONS.map((o) => {
          const active = o.value === value;
          return (
            <button
              key={o.value}
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(o.value)}
              className={[
                "rounded-lg px-3 py-2.5 text-left transition-colors",
                disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
                active
                  ? "bg-[var(--bg)] ring-1 ring-[var(--accent-dim)]"
                  : "hover:bg-[var(--bg)]/60",
              ].join(" ")}
            >
              <span
                className={`block text-sm font-medium ${active ? "text-[var(--accent)]" : "text-[var(--text-muted)]"}`}
              >
                {o.label}
              </span>
              <span className="mt-0.5 block text-[10px] leading-snug text-[var(--text-dim)]">
                {o.blurb}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
