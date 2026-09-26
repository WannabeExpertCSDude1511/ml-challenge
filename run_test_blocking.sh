#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================================="
echo "       RUNNING HEAVY BLOCKING TEST ON ENTIRE TRAINING DATASET           "
echo "========================================================================="

# 1. Check Python virtual environment
if [ -d ".venv" ]; then
    echo "[Step 1] Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "[Step 1] Virtual environment not found. Setting up .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
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

# 3. Execute blocking test on all S1 records (sample-size 0)
echo "[Step 2] Executing blocking test across full training dataset..."
python -m src.test_blocking --data "$DATA_DIR" --sample-size 0

echo ""
echo "========================================================================="
echo "                 FULL BLOCKING TEST RUN COMPLETE                         "
echo "========================================================================="
