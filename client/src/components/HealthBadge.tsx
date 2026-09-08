"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import type { HealthResult } from "@/lib/types";

/**
 * Live backend status in the header.
 *
 * Worth the space because the most common failure in this system is not a bad prediction -
 * it is the Python API not running, or running without a trained model. Both produce
 * confusing errors at upload time. Surfacing the state up front turns a mid-task failure
 * into something the user reads before they try.
 */
export function HealthBadge() {
  const [health, setHealth] = useState<HealthResult | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const h = await getHealth();
        if (!cancelled) {
          setHealth(h);
          setFailed(false);
        }
      } catch {
        if (!cancelled) {
          setHealth(null);
          setFailed(true);
        }
      }
    };

    check();
    // Poll slowly. The backend state changes only when someone starts it or finishes a
    // training run, so a tight interval would spend requests to learn nothing.
    const id = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const anyModel = health
    ? Object.values(health.models).some((m) => m.trained)
    : false;

  let tone = "var(--text-dim)";
  let label = "checking…";
  let title = "Contacting the API";

  if (failed) {
    tone = "var(--error)";
    label = "API offline";
    title = "The backend is not reachable. Start it with: ./run.sh serve";
  } else if (health) {
    if (!anyModel) {
      tone = "var(--danger)";
      label = "no model";
      title = "The API is up but no model has been trained yet.";
    } else {
      tone = "var(--safe)";
      label = health.device && health.device !== "cpu" ? `online · ${health.device}` : "online";
      title = `API v${health.version}${
        health.torch_available ? ` · torch on ${health.device}` : " · classical only"
      }`;
    }
  }

  return (
    <span
      title={title}
      className="ml-2 inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--bg-panel)] px-3 py-1.5 text-[11px] text-[var(--text-muted)]"
    >
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${failed || !health ? "animate-pulse-subtle" : ""}`}
        style={{ background: tone }}
      />
      {label}
    </span>
  );
}
