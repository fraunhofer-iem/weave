#!/bin/bash
#SBATCH -J "weave-shap-source"
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A hpc-prf-crnrw

# Point this at your checkout; every path below is derived from it.
export WEAVE_HOME="${WEAVE_HOME:-$PC2PFS/hpc-prf-crnrw/weave}"

# Which model to explain: "old" (the previously selected model) or "new".
# Submit once per model, e.g. sbatch --export=ALL,MODEL_TAG=new <script>
export MODEL_TAG="${MODEL_TAG:-old}"
case "$MODEL_TAG" in
  new) MODEL="weka.classifiers.trees.RandomForest -P 94 -I 100 -num-slots 1 -do-not-check-capabilities -K 0 -M 8.0 -V 0.001 -S 1" ;;
  old) MODEL="weka.classifiers.functions.SMO" ;;
  *) echo "MODEL_TAG must be old or new, got: $MODEL_TAG" >&2; exit 2 ;;
esac
echo "Explaining $MODEL_TAG model: $MODEL"
echo "Setting up WEAVE"

module purge
module load DefaultModules
module load lang/Java/17.0.6
module load lang/Python/3.12.3-GCCcore-13.3.0

echo "Setting up Python environment"
VENV="${WEAVE_HOME}/weave_env_so"
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

java -XX:MaxRAMPercentage=75 -jar ${WEAVE_HOME}/target/cli-1.0-jar-with-dependencies.jar -t weka -c ${WEAVE_HOME}/evaluation/ml4srm/swan/source.properties -X -e "$MODEL"

echo "Process completed"