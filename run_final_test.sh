#!/bin/bash
set -e
source .venv/bin/activate

echo "=========================================================="
echo "🎯 RUNNING FINAL PREDICTION & SUBMISSION (dataset/test)"
echo "=========================================================="
echo ""
echo "--- [1/2] Generating final predictions ---"
time python -m src.predict --data dataset/test --model model.joblib --output output

echo ""
echo "--- [2/2] Packaging submission zip ---"
./prepare_submission.sh

echo ""
echo "✅ Ready to submit! Check for the .zip file."
