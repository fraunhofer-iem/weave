#!/bin/bash
#SBATCH -J "weave-swan-cwe79"
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -A hpc-prf-crnrw

# Point this at your checkout; every path below is derived from it.
export ML4SRM_HOME="${ML4SRM_HOME:-$PC2PFS/hpc-prf-crnrw/ml4srm}"
echo "Setting up WEAVE"

module purge
module load DefaultModules
module load lang/Java/17.0.6

echo "Starting model selection"

java -jar ${ML4SRM_HOME}/weave-cli-1.0.jar -t weka -c ${ML4SRM_HOME}/swan/swan-cwe79.properties

echo "Model selected"