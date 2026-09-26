#!/bin/bash
set -e
source .venv/bin/activate

echo "=========================================================="
echo "🏋️ RUNNING FULL TRAINING PIPELINE (dataset/train)"
echo "=========================================================="
echo "⚠️ Running on the FULL dataset. This will require significant RAM (100GB+)."
echo ""
echo "--- [1/2] Training on full dataset ---"
time python -m src.train --data dataset/train --model model.joblib --model-type lightgbm

echo ""
echo "--- [2/2] Evaluating on full dataset ---"
time python -m src.test_train --data dataset/train --model model.joblib

echo ""
echo "✅ Full training complete! Model saved as 'model.joblib'."
