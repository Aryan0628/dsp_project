"use client";

import type { AnalysisResult } from "@/lib/types";

/**
 * The numbers behind the verdict: region geometry, anatomical location, and the measured
 * gain from the enhancement stage.
 *
 * The location block is the part that earns its place on the typing task. Those three
 * numbers are the geometric cues a radiologist uses to name a tumor type - distance from
 * the midline, depth inside the parenchyma, contact with the brain's edge - so displaying
 * them lets a reviewer check the model's reasoning against their own, rather than only its
 * conclusion.
 */
export function MeasurementPanel({ result }: { result: AnalysisResult }) {
  const { segmentation: seg, enhancement: enh, location } = result;
  const p = seg.properties;

  return (
    <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-panel)] p-5">
      <h2 className="mb-4 text-sm font-semibold">Measurements</h2>

      {/* ---- region geometry */}
      <Block title="Detected region">
        {!seg.tumor_found || !p ? (
          <p className="text-xs text-[var(--text-dim)]">
            No region passed the plausibility filters on this slice. That is itself
            evidence — the absence of a candidate is one of the strongest signals the
            classifier receives.
          </p>
        ) : (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
            <Row label="Area" value={`${p.area.toFixed(0)} px`} />
            <Row label="Share of brain" value={`${(p.area_ratio * 100).toFixed(2)}%`} />
            <Row label="Brightness vs brain" value={`${p.brightness_ratio.toFixed(2)}×`} />
            <Row
              label="Circularity"
              value={p.circularity.toFixed(3)}
              hint="1.0 = a perfect circle"
            />
            <Row
              label="Solidity"
              value={p.solidity.toFixed(3)}
              hint="area ÷ convex-hull area; low = ragged edge"
            />
            <Row label="Aspect ratio" value={p.aspect_ratio.toFixed(2)} />
            <Row
              label="Centroid"
              value={`${p.centroid[0].toFixed(0)}, ${p.centroid[1].toFixed(0)}`}
            />
            <Row label="Perimeter" value={`${p.perimeter.toFixed(0)} px`} />
            <Row label="Candidates" value={String(seg.n_candidates)} />
          </dl>
        )}
      </Block>

      {/* ---- anatomical location */}
      {location && location.loc_found === 1 && (
        <Block title="Anatomical location" note="the cues that separate tumor types">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
            <Row
              label="Distance from midline"
              value={location.loc_midline_abs?.toFixed(3) ?? "—"}
              hint="0 = on the midline (pituitary); higher = lateral"
            />
            <Row
              label="Depth in brain"
              value={location.loc_depth_centroid?.toFixed(3) ?? "—"}
              hint="1 = deep intra-axial (glioma); 0 = at the edge"
            />
            <Row
              label="Edge contact"
              value={
                location.loc_border_contact != null
                  ? `${(location.loc_border_contact * 100).toFixed(0)}%`
                  : "—"
              }
              hint="share hugging the brain border (meningioma)"
            />
            <Row
              label="Radial position"
              value={location.loc_radial?.toFixed(3) ?? "—"}
              hint="0 = brain centre, 1 = at the rim"
            />
          </dl>
        </Block>
      )}

      {/* ---- enhancement gain */}
      <Block title="Enhancement gain" note="Stage 3, measured rather than asserted">
        <dl className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-3">
          <Delta label="Contrast (std)" before={enh.contrast_before} after={enh.contrast_after} />
          <Delta
            label="Entropy (bits)"
            before={enh.entropy_before}
            after={enh.entropy_after}
            digits={2}
          />
          <Delta
            label="Sharpness"
            before={enh.sharpness_before}
            after={enh.sharpness_after}
          />
        </dl>
      </Block>
    </section>
  );
}

function Block({
  title,
  note,
  children,
}: {
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-5 last:mb-0">
      <p className="mb-2 text-[11px] uppercase tracking-wider text-[var(--text-dim)]">
        {title}
        {note && <span className="ml-2 normal-case tracking-normal">· {note}</span>}
      </p>
      {children}
    </div>
  );
}

function Row({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <dt className="text-[11px] text-[var(--text-dim)]" title={hint}>
        {label}
      </dt>
      <dd className="tabular text-sm text-[var(--text)]">{value}</dd>
      {hint && <p className="mt-0.5 text-[10px] leading-snug text-[var(--text-dim)]">{hint}</p>}
    </div>
  );
}

function Delta({
  label,
  before,
  after,
  digits = 1,
}: {
  label: string;
  before: number;
  after: number;
  digits?: number;
}) {
  const up = after > before;
  return (
    <div>
      <dt className="text-[11px] text-[var(--text-dim)]">{label}</dt>
      <dd className="tabular text-sm">
        <span className="text-[var(--text-dim)]">{before.toFixed(digits)}</span>
        <span className="mx-1.5 text-[var(--text-dim)]">→</span>
        <span style={{ color: up ? "var(--safe)" : "var(--text)" }}>{after.toFixed(digits)}</span>
      </dd>
    </div>
  );
}
