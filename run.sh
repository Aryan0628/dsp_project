#!/usr/bin/env bash
# Convenience wrapper so no one has to remember the venv path during a demo.
set -e
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Virtual environment not found. Creating it now (one-time, ~2 minutes) ..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip -q
    .venv/bin/python -m pip install -q -r requirements.txt
    echo "Environment ready."
fi

case "${1:-help}" in
    check)
        echo "Running pre-flight checks ..."
        if $PY -m pytest tests -q; then
            echo
            echo "  ALL CHECKS PASSED - the system is ready to demo."
        else
            echo
            echo "  CHECKS FAILED - see the output above."
            exit 1
        fi
        ;;
    train)     shift; exec $PY main.py train "$@" ;;
    train-all) exec ./scripts/train_all.sh ;;
    demo)      shift; exec $PY main.py demo "$@" ;;
    predict)   shift; exec $PY main.py predict "$@" ;;
    generate)  shift; exec $PY main.py generate "$@" ;;
    metrics)   shift; exec $PY main.py metrics "$@" ;;
    serve)     shift; exec $PY main.py serve "$@" ;;
    client)
        # The frontend is a separate npm project; install on first run so a fresh clone
        # only ever needs the two commands printed by `./run.sh help`.
        if [ ! -d client/node_modules ]; then
            echo "Installing client dependencies (one-time) ..."
            (cd client && npm install)
        fi
        exec npm --prefix client run dev
        ;;
    build-client)
        (cd client && npm install && npm run build)
        ;;
    setup)
        $PY -m pip install -q -r requirements.txt
        (cd client && npm install)
        echo "Dependencies installed (Python + client)."
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
    ./run.sh setup              (re)install dependencies, Python and client
HELP
        ;;
esac
