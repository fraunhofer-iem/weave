#!/bin/bash
#SBATCH -J "weave-shap-cwe79"
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A hpc-prf-crnrw
echo "Setting up WEAVE"

module purge
module load DefaultModules
module load lang/Java/17.0.6
module load lang/Python/3.12.3-GCCcore-13.3.0

echo "Setting up Python environment"
python -m venv weave_env_cwe79
source weave_env_cwe79/bin/activate

echo "Installing packages"
pip install numpy
pip install pandas
pip install shap
pip install matplotlib
pip install requests

echo "Starting model explanation"

java -Xms20g -Xmx200g -jar $PC2PFS/hpc-prf-crnrw/weave/target/cli-1.0-jar-with-dependencies.jar -t weka -c $PC2PFS/hpc-prf-crnrw/weave/evaluation/ml4srm/swan/cwe79.properties -X -e "weka.classifiers.meta.RandomSubSpace -P 0.83125 -S 1 -num-slots 1 -I 61 -W weka.classifiers.functions.Logistic -do-not-check-capabilities -- -R 1.9285573065288986 -M 10 -do-not-check-capabilities -num-decimal-places 4"

echo "Process completed"