#!/bin/bash
#SBATCH -J "weave-shap-cwe89"
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

java -Xms20g -Xmx200g -jar $PC2PFS/hpc-prf-crnrw/weave/target/cli-1.0-jar-with-dependencies.jar -t weka -c $PC2PFS/hpc-prf-crnrw/weave/evaluation/ml4srm/swan/cwe89.properties -X -e "weka.classifiers.lazy.LWL -U 2 -K 120 -A weka.core.neighboursearch.LinearNNSearch -W weka.classifiers.trees.RandomForest -do-not-check-capabilities -- -P 24 -I 118 -num-slots 1 -do-not-check-capabilities -K 32 -M 8.0 -V 10.0 -S 1 -N 2 -B"

echo "Process completed"