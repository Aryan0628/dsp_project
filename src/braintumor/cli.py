"""Command line interface. Every demo action is one subcommand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config


def _banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


# --------------------------------------------------------------------------- commands
def cmd_generate(args) -> int:
    from .dataset import generate_synthetic_dataset

    _banner(f"GENERATING {args.count} SYNTHETIC BRAIN MRI IMAGES")
    paths, labels = generate_synthetic_dataset(args.count, seed=args.seed)
    print(f"  tumor   : {sum(labels)} -> {config.SYNTHETIC_DIR / 'yes'}")
    print(f"  healthy : {len(labels) - sum(labels)} -> {config.SYNTHETIC_DIR / 'no'}")
    print(f"\nDone. {len(paths)} images written.")
    return 0


def cmd_train(args) -> int:
    from .train import train
    from .visualize import results_dashboard

    report = train(
        dataset=args.dataset,
        multiclass=args.multiclass,
        n_synthetic=args.count,
        seed=args.seed,
        limit=args.limit,
        deep=args.deep,
        deep_epochs=args.epochs,
    )
    suffix = "multiclass" if args.multiclass else "binary"
    out = config.OUTPUT_DIR / f"training_report_{suffix}.png"
    results_dashboard(report, out)
    print(f"  Dashboard    : {out}")
    return 0


def cmd_predict(args) -> int:
    from .predict import analyse_file, verdict_text
    from .visualize import pipeline_figure

    path = Path(args.image)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1

    _banner(f"ANALYSING  {path.name}")
    result = analyse_file(path)
    print(verdict_text(result))

    out = Path(args.output) if args.output else config.OUTPUT_DIR / f"result_{path.stem}.png"
    pipeline_figure(result, out)
    print(f"\nFigure saved: {out}")
    return 0


def cmd_demo(args) -> int:
    """One command that produces every figure needed for the presentation."""
    import random

    from .dataset import generate_brain_mri
    from .predict import analyse_image, analyse_file, verdict_text
    from .preprocess import load_image, preprocess
    from .train import load_model
    from .visualize import (
        enhancement_comparison,
        pipeline_figure,
        real_sample_grid,
        sample_grid,
    )

    _banner("DEMO - generating every presentation figure")
    try:
        bundle = load_model()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    outputs: list[Path] = []
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # ---- prefer REAL scans; fall back to the simulator when no dataset is installed
    cases: list[tuple[str, object]] = []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for kind, folder in (("tumor", config.RAW_DIR / "yes"), ("healthy", config.RAW_DIR / "no")):
        if folder.is_dir():
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
            if files:
                cases.append((kind, random.choice(files)))

    if not cases:
        print("  no real dataset found - using synthetic phantoms")
        for kind, is_tumor in (("tumor", True), ("healthy", False)):
            raw = generate_brain_mri(is_tumor, rng)
            p = config.OUTPUT_DIR / f"demo_{kind}_input.png"
            cv2.imwrite(str(p), raw)
            cases.append((kind, p))
            outputs.append(p)

    for kind, path in cases:
        result = analyse_file(path, bundle)
        print(f"\n--- {kind.upper()} case: {Path(path).name} " + "-" * 30)
        print(verdict_text(result))
        fig_path = config.OUTPUT_DIR / f"demo_{kind}_pipeline.png"
        pipeline_figure(result, fig_path)
        outputs.append(fig_path)

    # ---- Stage 3 comparison, on a real scan when one is available
    ref_src = cases[0][1]
    ref = preprocess(load_image(ref_src)) if not isinstance(ref_src, np.ndarray) else preprocess(ref_src)
    outputs.append(enhancement_comparison(ref, config.OUTPUT_DIR / "demo_enhancement.png"))

    # ---- grids
    grid = real_sample_grid(config.OUTPUT_DIR / "demo_samples.png", bundle, n=5, seed=args.seed)
    if grid is None:
        grid = sample_grid(config.OUTPUT_DIR / "demo_samples.png", n=4, seed=args.seed)
    outputs.append(grid)

    print("\n" + "-" * 74)
    print("Figures written:")
    for p in outputs:
        print(f"  {p}")
    return 0


def cmd_metrics(args) -> int:
    task = "multiclass" if getattr(args, "multiclass", False) else "binary"
    path = config.OUTPUT_DIR / f"metrics_{task}.json"
    if not path.exists():
        path = config.METRICS_PATH
    if not path.exists():
        print("No metrics yet. Run:  python main.py train", file=sys.stderr)
        return 1
    from .train import _print_report

    _print_report(json.loads(path.read_text()))
    return 0


def cmd_serve(args) -> int:
    """Run the REST API that the Next.js client talks to."""
    import uvicorn

    print(f"Starting the API on http://{args.host}:{args.port}")
    print(f"  interactive docs : http://{args.host}:{args.port}/docs")
    print("  Ctrl+C to stop.\n")
    uvicorn.run(
        "braintumor.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


# ------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Brain Tumor Detection from MRI - OpenCV enhancement + ML classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py train                       train on the best real dataset available
  python main.py train --dataset classic     253-image binary benchmark
  python main.py train --dataset kaggle4     3264-image dataset, provided train/test split
  python main.py train --dataset kaggle4 --multiclass
                                             classify tumor TYPE, not just presence
  python main.py train --dataset kaggle4 --multiclass --deep
                                             ...and add the fine-tuned CNN branch
  python main.py train --dataset synthetic   generated phantoms, no download needed
  python main.py predict scan.png            analyse one MRI, save the 8-panel figure
  python main.py demo                        produce every presentation figure
  python main.py metrics [--multiclass]      reprint the last training report
  python main.py serve                       run the REST API for the web client
""",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create the synthetic MRI dataset")
    g.add_argument("--count", type=int, default=config.SYNTH_DEFAULT_COUNT)
    g.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train and evaluate the classifier")
    t.add_argument("--dataset", default="auto",
                   choices=["auto", "classic", "kaggle4", "synthetic"],
                   help="auto = largest real dataset available; classic = 253-image binary "
                        "set; kaggle4 = 3264-image 4-class set; synthetic = generated")
    t.add_argument("--multiclass", action="store_true",
                   help="classify tumor TYPE (glioma/meningioma/pituitary/none) instead of "
                        "just present/absent. Requires --dataset kaggle4.")
    t.add_argument("--count", type=int, default=config.SYNTH_DEFAULT_COUNT,
                   help="number of synthetic images when --dataset synthetic")
    t.add_argument("--limit", type=int, default=None,
                   help="use only N training images (for a quick smoke run)")
    t.add_argument("--deep", action="store_true",
                   help="also fine-tune a ResNet-18 and ship the better of it, the "
                        "classical model, or a blend. Mainly for --multiclass, where the "
                        "engineered features were measured to fall short.")
    t.add_argument("--epochs", type=int, default=config.DEEP_EPOCHS,
                   help="training epochs for the --deep branch")
    t.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="analyse a single MRI image")
    pr.add_argument("image")
    pr.add_argument("-o", "--output")
    pr.set_defaults(func=cmd_predict)

    d = sub.add_parser("demo", help="generate all presentation figures")
    d.add_argument("--seed", type=int, default=5)
    d.set_defaults(func=cmd_demo)

    m = sub.add_parser("metrics", help="reprint the last training report")
    m.add_argument("--multiclass", action="store_true",
                   help="show the multi-class report instead of the binary one")
    m.set_defaults(func=cmd_metrics)

    s = sub.add_parser("serve", help="run the REST API used by the Next.js client")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=config.API_PORT)
    s.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
