#!/usr/bin/env bash
# Convenience wrapper so no one has to remember the venv path during a demo.
set -e
cd "$(dirname "$0")"

# Layout: backend/ is the Python ML API, frontend/ is the React + TypeScript web app.
# They only talk over HTTP, so each can be deployed on its own (see README "Deployment").
PY="$PWD/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Virtual environment not found. Creating it now (one-time, ~2 minutes) ..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip -q
    .venv/bin/python -m pip install -q -r backend/requirements.txt
    echo "Environment ready."
fi

# Every Python command below runs from inside backend/.
run_py() { (cd backend && exec "$PY" "$@"); }

case "${1:-help}" in
    check)
        echo "Running pre-flight checks ..."
        if run_py -m pytest tests -q; then
            echo
            echo "  ALL CHECKS PASSED - the system is ready to demo."
        else
            echo
            echo "  CHECKS FAILED - see the output above."
            exit 1
        fi
        ;;
    train)     shift; run_py main.py train "$@" ;;
    train-all) exec ./backend/scripts/train_all.sh ;;
    demo)      shift; run_py main.py demo "$@" ;;
    predict)
        shift
        # Python runs inside backend/, so turn an image path given from here into an
        # absolute one first (a path already relative to backend/ also works).
        if [ -n "$1" ] && [ -e "$1" ]; then set -- "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" "${@:2}"; fi
        run_py main.py predict "$@" ;;
    generate)  shift; run_py main.py generate "$@" ;;
    metrics)   shift; run_py main.py metrics "$@" ;;
    serve)     shift; run_py main.py serve "$@" ;;
    client|frontend)
        # The frontend is a separate npm project; install on first run so a fresh clone
        # only ever needs the two commands printed by `./run.sh help`.
        if [ ! -d frontend/node_modules ]; then
            echo "Installing frontend dependencies (one-time) ..."
            (cd frontend && npm install)
        fi
        exec npm --prefix frontend run dev
        ;;
    build-client)
        (cd frontend && npm install && npm run build)
        ;;
    setup)
        "$PY" -m pip install -q -r backend/requirements.txt
        (cd frontend && npm install)
        echo "Dependencies installed (Python + frontend)."
        ;;
    *)
        cat <<'HELP'
Brain Tumor Detection and Typing from MRI - command shortcuts

  RUN THE WEB APP  (two terminals)
    ./run.sh serve              start the REST API          -> http://localhost:3000
    ./run.sh client             start the Next.js frontend  -> http://localhost:5713

  TRAIN
    ./run.sh train-all          retrain both shipped models (binary + typing w/ CNN)
    ./run.sh train              binary detection on the best dataset available
    ./run.sh train --dataset kaggle4 --multiclass --deep
                                tumor typing, classical + fine-tuned CNN ensemble

  EVERYTHING ELSE
    ./run.sh check              run the test suite (verify everything works)
    ./run.sh demo               generate every presentation figure
    ./run.sh predict <image>    analyse one MRI image
    ./run.sh metrics [--multiclass]   reprint the last training report
    ./run.sh generate           regenerate the synthetic dataset
    ./run.sh setup              (re)install dependencies, Python and frontend
HELP
        ;;
esac
