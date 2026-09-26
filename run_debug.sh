#!/bin/bash
set -e
source .venv/bin/activate

echo "=========================================================="
echo "🧪 RUNNING DEBUG PIPELINE (dataset/debug)"
echo "=========================================================="
echo ""
echo "--- [1/2] Training on debug data ---"
time python -m src.train --data dataset/debug --model debug_model.joblib --model-type lightgbm

echo ""
echo "--- [2/2] Evaluating on debug data ---"
time python -m src.test_train --data dataset/debug --model debug_model.joblib

echo ""
echo "✅ Debug run complete!"
