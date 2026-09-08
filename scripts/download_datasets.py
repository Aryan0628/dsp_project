#!/usr/bin/env python3
"""
Download the real brain MRI datasets this project trains on.

Both are public and are fetched from Hugging Face, which needs no account or API key
(unlike Kaggle). Run from the project root:

    .venv/bin/python scripts/download_datasets.py            # both datasets
    .venv/bin/python scripts/download_datasets.py --only classic
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / ".download_cache"

HF = "https://huggingface.co/datasets"

DATASETS = {
    "classic": {
        "name": "Brain MRI Images for Brain Tumor Detection (Chakrabarty)",
        "detail": "253 images, binary yes/no. The standard small benchmark for this task.",
        "urls": [f"{HF}/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection/resolve/main/"
                 "Brain%20MRI%20Images%20for%20Brain%20Tumor%20Detection.zip"],
        "target": DATA / "raw",
    },
    "kaggle4": {
        "name": "Brain Tumor Classification MRI (Bhuvaji et al.)",
        "detail": "3264 images in 4 classes with a provided train/test split.",
        "urls": [f"{HF}/sartajbhuvaji/Brain-Tumor-Classification/resolve/main/Training.zip",
                 f"{HF}/sartajbhuvaji/Brain-Tumor-Classification/resolve/main/Testing.zip"],
        "target": DATA / "kaggle4",
    },
}

CLASS_DIRS = ("glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor")


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    cached: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    downloading {dest.name} ...", flush=True)

    # Only animate the progress meter on a real terminal. Redirected to a file or a pipe,
    # the carriage returns pile up into thousands of useless lines.
    interactive = sys.stdout.isatty()
    state = {"last": -1}

    def hook(block, block_size, total):
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = int(done / total * 100)
        if interactive:
            print(f"\r      {pct:3d}%  ({done/1e6:.1f}/{total/1e6:.1f} MB)",
                  end="", flush=True)
        elif pct >= state["last"] + 25:            # coarse milestones when logging
            state["last"] = pct
            print(f"      {pct:3d}%", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    if interactive:
        print()
    return dest


def install_classic(zips: list[Path], target: Path) -> None:
    """The archive contains brain_tumor_dataset/{yes,no} plus a duplicate copy at the top
    level. We take only the brain_tumor_dataset/ copy, so images are not counted twice."""
    work = CACHE / "classic_extract"
    if work.exists():
        shutil.rmtree(work)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(work)

    src = work / "brain_tumor_dataset"
    if not src.is_dir():
        raise RuntimeError(f"unexpected archive layout under {work}")

    for cls in ("yes", "no"):
        out = target / cls
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted((src / cls).iterdir()):
            if f.is_file():
                shutil.copy2(f, out / f.name)
        print(f"    {cls}: {len(list(out.iterdir()))} images -> {out}")


def install_kaggle4(zips: list[Path], target: Path) -> None:
    work = CACHE / "kaggle4_extract"
    if work.exists():
        shutil.rmtree(work)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(work)

    for src_name, dst_name in (("Training", "train"), ("Testing", "test")):
        src = work / src_name
        if not src.is_dir():
            raise RuntimeError(f"missing {src_name}/ in the archive")
        for cls in CLASS_DIRS:
            out = target / dst_name / cls
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in sorted((src / cls).iterdir()):
                if f.is_file():
                    shutil.copy2(f, out / f.name)
                    n += 1
            print(f"    {dst_name}/{cls}: {n} images")


INSTALLERS = {"classic": install_classic, "kaggle4": install_kaggle4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(DATASETS), help="fetch just one dataset")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the downloaded zips (default: delete after install)")
    args = ap.parse_args()

    wanted = [args.only] if args.only else list(DATASETS)
    CACHE.mkdir(parents=True, exist_ok=True)

    for key in wanted:
        spec = DATASETS[key]
        print(f"\n[{key}] {spec['name']}")
        print(f"    {spec['detail']}")
        try:
            zips = [fetch(u, CACHE / f"{key}_{i}.zip") for i, u in enumerate(spec["urls"])]
            INSTALLERS[key](zips, spec["target"])
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            print("    You can install it by hand instead - see docs/06_DATASETS.md",
                  file=sys.stderr)
            return 1

    if not args.keep_cache and CACHE.exists():
        shutil.rmtree(CACHE)

    print("\nDone. Now run:")
    print("    ./run.sh train --dataset kaggle4      # the main model")
    print("    ./run.sh train --dataset classic      # the small benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
