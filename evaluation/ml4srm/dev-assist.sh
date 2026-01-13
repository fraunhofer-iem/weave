#!/bin/bash
#SBATCH -J "weave-shap-dev-assist"
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A hpc-prf-crnrw
echo "Setting up WEAVE"

module purge
module load DefaultModules
module load lang/Java/17.0.6
module load lang/Python/3.12.3-GCCcore-13.3.0

echo "Setting up Python environment"
python -m venv weave_env
source weave_env/bin/activate

echo "Installing packages"
pip install numpy
pip install pandas
pip install shap
pip install matplotlib
pip install requests

echo "Starting model explanation"

java -Xms20g -Xmx200g -jar $PC2PFS/hpc-prf-crnrw/weave/target/cli-1.0-jar-with-dependencies.jar -t weka -c $PC2PFS/hpc-prf-crnrw/weave/evaluation/ml4srm/dev-assist/dev-assist.properties -X -e "meka.classifiers.multilabel.meta.EnsembleML -S 1 -I 30 -P 64 -W meka.classifiers.multilabel.PS -- -P 4 -N 4 -S 0 -W weka.classifiers.trees.J48 -- -C 0.4375 -B -M 2"

echo "Process completed"