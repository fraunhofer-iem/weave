#!/bin/bash
#SBATCH -J "weave-shap-dev-assist"
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A hpc-prf-crnrw

# Point this at your checkout; every path below is derived from it.
export WEAVE_HOME="${WEAVE_HOME:-$PC2PFS/hpc-prf-crnrw/weave}"

# Which model to explain: "old" (the previously selected model) or "new".
# Submit once per model, e.g. sbatch --export=ALL,MODEL_TAG=new <script>
export MODEL_TAG="${MODEL_TAG:-old}"
case "$MODEL_TAG" in
  new) MODEL="meka.classifiers.multilabel.meta.EnsembleML -S 1 -I 30 -P 64 -W meka.classifiers.multilabel.PS -- -P 4 -N 4 -S 0 -W weka.classifiers.trees.J48 -- -C 0.4375 -B -M 2" ;;
  old) MODEL="meka.classifiers.multilabel.meta.EnsembleML -S 1 -I 10 -P 67 -W meka.classifiers.multilabel.PS -- -P 0 -N 0 -S 0 -W weka.classifiers.trees.LMT -- -I -1 -M 15 -W 00" ;;
  *) echo "MODEL_TAG must be old or new, got: $MODEL_TAG" >&2; exit 2 ;;
esac
echo "Explaining $MODEL_TAG model: $MODEL"
echo "Setting up WEAVE"

module purge
module load DefaultModules
module load lang/Java/17.0.6
module load lang/Python/3.12.3-GCCcore-13.3.0

echo "Setting up Python environment"
python -m venv weave_env_dev_assist
source weave_env_dev_assist/bin/activate

echo "Installing packages"
pip install numpy
pip install pandas
pip install shap
pip install matplotlib
pip install requests

echo "Starting model explanation"

java -Xms20g -Xmx200g -jar ${WEAVE_HOME}/target/cli-1.0-jar-with-dependencies.jar -t meka -c ${WEAVE_HOME}/evaluation/ml4srm/dev-assist/dev-assist.properties -X -e "$MODEL"

echo "Process completed"