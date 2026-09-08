"use client";

import { useCallback, useRef, useState } from "react";

const ACCEPTED = ["image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"];
const MAX_BYTES = 20 * 1024 * 1024;

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
  filename?: string | null;
}

/**
 * Drag-and-drop / click-to-browse MRI upload.
 *
 * Validation is duplicated here and in the backend on purpose. The client copy exists to
 * give an instant, specific message without a round trip; the server copy exists because a
 * client check is advice, not a control - anything can POST to the API directly.
 */
export function UploadZone({ onFile, disabled, filename }: Props) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (file.size > MAX_BYTES) {
        setError(`That file is ${(file.size / 1e6).toFixed(1)} MB. The limit is 20 MB.`);
        return;
      }
      // Some browsers report an empty type for .tif/.bmp, so fall back to the extension
      // rather than rejecting a valid scan on a missing MIME string.
      const byExt = /\.(png|jpe?g|bmp|tiff?|webp)$/i.test(file.name);
      if (file.type && !ACCEPTED.includes(file.type) && !byExt) {
        setError(`"${file.type}" is not a supported image type.`);
        return;
      }
      setError(null);
      onFile(file);
    },
    [onFile],
  );

  return (
    <div>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload a brain MRI slice"
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files?.[0]);
        }}
        className={[
          "rounded-xl border-2 border-dashed p-8 text-center transition-colors",
          disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
          dragging
            ? "border-[var(--accent)] bg-[color-mix(in_srgb,var(--accent)_8%,transparent)]"
            : "border-[var(--border-strong)] bg-[var(--bg-panel)] hover:border-[var(--accent-dim)]",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",") + ",.tif,.tiff,.bmp"}
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            accept(e.target.files?.[0]);
            // Reset so choosing the same file twice still fires a change event - otherwise
            // re-running an analysis on the same scan silently does nothing.
            e.target.value = "";
          }}
        />

        <svg
          className="mx-auto mb-3"
          width="34"
          height="34"
          viewBox="0 0 24 24"
          fill="none"
          stroke={dragging ? "var(--accent)" : "var(--text-dim)"}
          strokeWidth="1.4"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>

        <p className="text-sm font-medium">
          {filename ? (
            <span className="text-[var(--accent)]">{filename}</span>
          ) : (
            <>Drop a brain MRI slice here, or click to browse</>
          )}
        </p>
        <p className="mt-1.5 text-xs text-[var(--text-dim)]">
          PNG · JPEG · BMP · TIFF · up to 20&nbsp;MB · a single 2D slice
        </p>
      </div>

      {error && (
        <p className="mt-2 text-xs text-[var(--error)]" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
