#!/bin/bash
#SBATCH -J "weave-shap-dev-assist"
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A hpc-prf-crnrw

# Point this at your checkout; every path below is derived from it.
export WEAVE_HOME="${WEAVE_HOME:-$PC2PFS/hpc-prf-crnrw/weave}"

# Which model to explain: "published" (the model the tool shipped with) or "weave"
# (the model ML-Plan selected in stage 1). These are the names the paper uses.
# Submit once per model, e.g. sbatch --export=ALL,MODEL_TAG=weave <script>
export MODEL_TAG="${MODEL_TAG:-published}"
case "$MODEL_TAG" in
  weave) MODEL="meka.classifiers.multilabel.meta.EnsembleML -S 1 -I 30 -P 64 -W meka.classifiers.multilabel.PS -- -P 4 -N 4 -S 0 -W weka.classifiers.trees.J48 -- -C 0.4375 -B -M 2" ;;
  published) MODEL="meka.classifiers.multilabel.meta.EnsembleML -S 1 -I 10 -P 67 -W meka.classifiers.multilabel.PS -- -P 0 -N 0 -S 0 -W weka.classifiers.trees.LMT -- -I -1 -M 15 -W 00" ;;
  *) echo "MODEL_TAG must be published or weave, got: $MODEL_TAG" >&2; exit 2 ;;
esac
echo "Explaining $MODEL_TAG model: $MODEL"
echo "Setting up WEAVE"

module purge
module load DefaultModules
module load lang/Java/17.0.6
module load lang/Python/3.12.3-GCCcore-13.3.0

echo "Setting up Python environment"
VENV="${WEAVE_HOME}/weave_env_dev_assist"
REQS="${WEAVE_HOME}/evaluation/ml4srm/requirements.txt"

# The sentinel marks a fully provisioned venv. If it is absent the venv is
# rebuilt: that repairs environments left half-installed by a failed pip run.
if [ -f "$VENV/.weave-deps-ok" ]; then
  echo "Reusing existing environment $VENV"
  source "$VENV/bin/activate"
else
  echo "Building environment $VENV"
  rm -rf "$VENV"
  python -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --quiet --upgrade pip
  # One invocation, pinned: separate unpinned installs resolve independently and
  # end up with a numba/NumPy combination that cannot be imported.
  pip install --quiet -r "$REQS" && touch "$VENV/.weave-deps-ok"
fi

if ! python -c "import shap, numpy, pandas, matplotlib, requests" 2>/dev/null; then
  echo "ERROR: python dependencies are not importable in $VENV" >&2
  python -c "import shap" || true
  exit 1
fi

echo "Starting model explanation"

java -XX:MaxRAMPercentage=75 -jar ${WEAVE_HOME}/target/cli-1.0-jar-with-dependencies.jar -t meka -c ${WEAVE_HOME}/evaluation/ml4srm/dev-assist/dev-assist.properties -X -e "$MODEL"

echo "Process completed"