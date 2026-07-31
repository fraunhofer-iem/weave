#!/bin/bash
#SBATCH -J "weave-shap-cwe89"
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A hpc-prf-crnrw

# Point this at your checkout; every path below is derived from it.
export WEAVE_HOME="${WEAVE_HOME:-$PC2PFS/hpc-prf-crnrw/weave}"

# Which model to explain: "old" (the previously selected model) or "new".
# Submit once per model, e.g. sbatch --export=ALL,MODEL_TAG=new <script>
export MODEL_TAG="${MODEL_TAG:-old}"
case "$MODEL_TAG" in
  new) MODEL="weka.classifiers.functions.SimpleLogistic -I 0 -S -P -M 500 -H 32 -W 0.0 -do-not-check-capabilities" ;;
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
python -m venv weave_env_cwe89
source weave_env_cwe89/bin/activate

echo "Installing packages"
pip install numpy
pip install pandas
pip install shap
pip install matplotlib
pip install requests

echo "Starting model explanation"

java -Xms20g -Xmx200g -jar ${WEAVE_HOME}/target/cli-1.0-jar-with-dependencies.jar -t weka -c ${WEAVE_HOME}/evaluation/ml4srm/swan/cwe89.properties -X -e "$MODEL"

echo "Process completed"