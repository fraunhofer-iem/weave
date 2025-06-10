import argparse
import logging
import os

import numpy as np
import pandas as pd
import shap
import weka.core.converters as con
import weka.core.jvm as jvm
from matplotlib import pyplot as plt
from weka.classifiers import Classifier, Evaluation
from weka.core import converters

from Converters import inst_to_np

np.random.seed(42)
label = 'CWE89'

# Argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', required=True, help='Path to training ARFF file')
parser.add_argument('--test', required=True, help='Path to testing ARFF file')
parser.add_argument('--out', required=True, help='Output directory to store results')
args = parser.parse_args()

# Create output directories
output_path = f'{args.out}/results/'
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, f'local/csv/{label}'), exist_ok=True)

# Configure logger
logfile_path = f'{args.out}/logs'
os.makedirs(logfile_path, exist_ok=True)
logfile_path = os.path.join(logfile_path, f'{label}.log')
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logfile_path, format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filemode='w', encoding='utf-8', level=logging.DEBUG, force=True)
print('logs saved at:' + logfile_path)
logging.getLogger("shap").setLevel(logging.CRITICAL)
logging.getLogger("weka").setLevel(logging.CRITICAL)

# Start JVM
jvm.start(system_cp=True, packages=True, max_heap_size="12288m")

# Load dataset
dataset = con.load_any_file(args.train)
dataset.class_is_last()
logger.info(f'Dataset loaded...')

# Build the Classifier
classifier = Classifier(classname="weka.classifiers.lazy.LWL", options=[
    "-U", "0", "-K", "60", "-A", "weka.core.neighboursearch.CoverTree", "-W", "weka.classifiers.trees.RandomTree",
    "--", "-K", "0", "-M", "3.0", "-V", "100.0", "-S", "1", "-depth", "10", "-U", "-do-not-check-capabilities"
])
logger.info('Classifier = ' + classifier.__str__())

# Train classifier
classifier.build_classifier(dataset)
logger.info('Model training completed successfully...')

feature_names = dataset.attribute_names()

def weka_predict(input_data):
    """ Convert numpy array to Instances and return the classifier's class distribution for each instance"""
    weka_data = con.ndarray_to_instances(input_data, 'weka', 'Att-!', feature_names)
    weka_data.class_is_last()
    predictions = []
    for inst in weka_data:
        pred = classifier.distribution_for_instance(inst)
        predictions.append(pred)
    return np.array(predictions)

# Convert dataset to NumPy array for Shap Explainer
dataset_np = inst_to_np(dataset)

# Compute SHAP values
global_explainer = shap.Explainer(weka_predict, dataset_np, feature_names=feature_names, seed=42)
global_exp = global_explainer(dataset_np)

# Plot SHAP beeswarm plot
fig = plt.figure()
shap.plots.beeswarm(global_exp[:, :dataset.num_attributes - 1,1], show=False)
plt.xlabel('SHAP Value')
plt.savefig(f'{output_path}/{label}-global.pdf', bbox_inches='tight', dpi=300)
plt.close()
logger.info(f"Saved SHAP plot for {label}")

# Log all stats from the explainer
logger.info("--------------- Global Statistics ---------------")
global_shap_df = pd.DataFrame(global_exp[:, :, 1].values, columns=feature_names)
logger.info("Shap Values: \n" + global_shap_df.abs().mean().sort_values(ascending=False).to_string())
logger.info("Shap Summary: \n" + str(global_shap_df.describe()))

# Load dataset for local explainability
test_dataset = converters.load_any_file(args.test)
test_dataset.class_is_last()

# Evaluating the test set
evaluation = Evaluation(dataset)
evaluation.test_model(classifier, test_dataset)
logger.info('Evaluation output:' + evaluation.predictions.__str__())
logger.info('Evaluation for ' + label + ': \n' + evaluation.summary())
logger.info('F-Score: ' + evaluation.f_measure(0).__str__())
logger.info('F-Score 1: ' + evaluation.f_measure(1).__str__())

# Convert to NumPy array
test_dataset_np = inst_to_np(test_dataset)

# Compute SHAP values for test instances
local_explainer = shap.Explainer(weka_predict, dataset_np, feature_names=feature_names, seed=42)
local_exp = local_explainer(test_dataset_np)

# Plot SHAP force plots for local explainability
for i in range(0, len(test_dataset_np)):
    shap.plots.waterfall(local_exp[i,:dataset.num_attributes - 1, 1], show=False)
    plt.savefig(f'{output_path}/local/{label}-local-instance-{i}.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved SHAP force plot for instance {i}")

    # Save local shap values to csv files
    instance_csv_path = f'{output_path}/local/csv/{label}/instance_{i}.csv'
    local_shap_df = pd.DataFrame([local_exp[i,:, 1].values], columns=feature_names)
    local_shap_df = local_shap_df.T.reset_index()
    local_shap_df.columns = ['Feature', 'SHAP Value']
    local_shap_df.sort_values(by='SHAP Value', ascending=False).to_csv(instance_csv_path, index=False)

# Log all stats from the explainer
logger.info("--------------- Local Statistics ---------------")
local_shap_df = pd.DataFrame(local_exp[:, :, 1].values, columns=feature_names)
logger.info("Shap Values: \n" + local_shap_df.mean().sort_values(ascending=False).to_string())
logger.info("Shap Summary: \n" + str(local_shap_df.describe()))

# Stop JVM
jvm.stop()