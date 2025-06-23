#!/bin/bash
# Run the explainability pipeline for RQ2 and RQ3
set -e

# Set the virtual environment directory
VENV_DIR="./venv"

# Activate the virtual environment
if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
fi

# Run the programs for rq2 and 3
echo "Starting Weka-Wrapper..."

python shap/weka/weka_wrapper.py --approach swan --out ./evaluation/demo &
python shap/weka/weka_wrapper.py --approach sscm --out ./evaluation/demo &
python shap/meka/Dev-Assist.py --train ./evaluation/ml4srm/dev-assist/train/meka-code.arff --test ./evaluation/ml4srm/dev-assist/test/dev-assist-owasp-benchmark.arff --out ./evaluation/demo &

wait

echo "Completed..."

# Deactivate virtual environment
if [ -d "$VENV_DIR" ]; then
  deactivate
fi