#!/bin/bash
#SBATCH -J "weave-shap-sscm"
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A hpc-prf-crnrw
echo "Setting up WEAVE"

module purge
module load DefaultModules
module load lang/Java/17.0.6
module load lang/Python/3.12.3-GCCcore-13.3.0

echo "Setting up Python environment"
python -m venv weave_env_ss
source weave_env_ss/bin/activate

echo "Installing packages"
pip install numpy
pip install pandas
pip install shap
pip install matplotlib
pip install requests

echo "Starting model explanation"

java -Xms20g -Xmx200g -jar $PC2PFS/hpc-prf-crnrw/weave/target/cli-1.0-jar-with-dependencies.jar -t weka -c $PC2PFS/hpc-prf-crnrw/weave/evaluation/ml4srm/sscm/sscm.properties -X -e "weka.classifiers.meta.Bagging -P 179 -S 1 -num-slots 1 -I 28 -W weka.classifiers.trees.RandomForest -do-not-check-capabilities -- -P 100 -I 122 -num-slots 1 -do-not-check-capabilities -K 0 -M 1.0 -V 0.001 -S 1"

echo "Process completed"