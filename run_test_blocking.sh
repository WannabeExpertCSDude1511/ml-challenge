#!/usr/bin/env bash
# Blocking evaluation (recall and candidates per S1) on the training data.
#
#   ./run_test_blocking.sh        # all ~2.2M training S1 records (slow: ~3 hours)
#   ./run_test_blocking.sh 10000  # a seeded sample of S1 records (~1 minute)
set -e

SAMPLE_SIZE="${1:-0}"

# Change to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================================="
if [ "$SAMPLE_SIZE" = "0" ]; then
    echo "       RUNNING HEAVY BLOCKING TEST ON ENTIRE TRAINING DATASET           "
else
    echo "       RUNNING BLOCKING TEST ON $SAMPLE_SIZE TRAINING S1 RECORDS        "
fi
echo "========================================================================="

# Python 3 is "python3" on Linux/macOS and usually "python" on Windows.
PYTHON=python3
if ! "$PYTHON" -c "" >/dev/null 2>&1; then
    PYTHON=python
fi

# 1. Check Python virtual environment (bin/ on Linux/macOS, Scripts/ on Windows)
activate_venv() {
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        source .venv/Scripts/activate
    fi
}
if [ -d ".venv" ]; then
    echo "[Step 1] Activating virtual environment (.venv)..."
    activate_venv
else
    echo "[Step 1] Virtual environment not found. Setting up .venv..."
    "$PYTHON" -m venv .venv
    activate_venv
    echo "Installing required dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# 2. Verify dataset directory exists
DATA_DIR="dataset/train"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Training dataset directory '$DATA_DIR' not found."
    echo "Please place your challenge data under dataset/train/"
    exit 1
fi

# 3. Execute blocking test (sample size 0 = all S1 records)
echo "[Step 2] Executing blocking test (sample size $SAMPLE_SIZE, 0 = all)..."
python -m src.test_blocking --data "$DATA_DIR" --sample-size "$SAMPLE_SIZE"

echo ""
echo "========================================================================="
echo "                 BLOCKING TEST RUN COMPLETE                              "
echo "========================================================================="
