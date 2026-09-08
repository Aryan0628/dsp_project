import type { Metadata } from "next";
import "./globals.css";
import { HealthBadge } from "@/components/HealthBadge";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Brain MRI Diagnostic Assistance",
  description:
    "OpenCV enhancement, automated segmentation and machine-learning classification of "
    + "brain MRI slices - tumor detection and tumor typing. Assistive tool, not a medical device.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-[var(--border)] bg-[var(--bg-elevated)] sticky top-0 z-20">
            <div className="mx-auto max-w-7xl px-6 h-16 flex items-center gap-6">
              <Link href="/" className="flex items-center gap-3 group">
                <BrainMark />
                <div className="leading-tight">
                  <div className="font-semibold text-[15px] tracking-tight group-hover:text-[var(--accent)] transition-colors">
                    Brain MRI Diagnostic Assistance
                  </div>
                  <div className="text-[11px] text-[var(--text-dim)]">
                    Enhancement · Segmentation · Classification
                  </div>
                </div>
              </Link>

              <nav className="ml-auto flex items-center gap-1 text-sm">
                <NavLink href="/">Analyse</NavLink>
                <NavLink href="/performance">Performance</NavLink>
                <HealthBadge />
              </nav>
            </div>
          </header>

          <main className="flex-1">{children}</main>

          {/* The disclaimer is in the layout, not on one page. A tool that outputs a
              clinical-sounding verdict should never render a screen that omits it. */}
          <footer className="border-t border-[var(--border)] bg-[var(--bg-elevated)] mt-16">
            <div className="mx-auto max-w-7xl px-6 py-5 text-xs text-[var(--text-dim)] flex flex-wrap gap-x-6 gap-y-2 justify-between">
              <p className="max-w-3xl">
                <strong className="text-[var(--text-muted)]">Assistive tool only.</strong>{" "}
                This is a decision-support and triage prototype. It is not a medical device,
                has no regulatory approval, and must never be the sole basis for a clinical
                decision. A qualified radiologist makes the final call.
              </p>
              <p className="tabular">Reproducible run · fixed seed 42</p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="px-3 py-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-panel)] transition-colors"
    >
      {children}
    </Link>
  );
}

function BrainMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <circle cx="16" cy="16" r="13" stroke="var(--accent)" strokeWidth="1.5" opacity="0.55" />
      <path
        d="M16 4.5v23M8 9c3 2.2 3 5.4 0 7.6M24 9c-3 2.2-3 5.4 0 7.6"
        stroke="var(--accent)"
        strokeWidth="1.2"
        strokeLinecap="round"
        opacity="0.65"
      />
      <circle cx="20.5" cy="12.5" r="3.2" fill="var(--danger)" opacity="0.85" />
    </svg>
  );
}
