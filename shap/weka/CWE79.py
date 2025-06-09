import logging
import os

import numpy as np
import pandas as pd
import shap
import weka.core.jvm as jvm
import weka.core.converters as converters
from matplotlib import pyplot as plt
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from Converters import inst_to_np
import weka.core.converters as con

np.random.seed(42)
label = 'CWE79'

# Configure logger
logfile_path = os.getcwd() + f'/out/{label}.log'
logger = logging.getLogger(label)
logging.basicConfig(filename=logfile_path, format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filemode='w', encoding='utf-8', level=logging.INFO, force=True)
print('logs saved at:' + logfile_path)
logging.getLogger("shap").setLevel(logging.CRITICAL)
logging.getLogger("weka").setLevel(logging.CRITICAL)

# Start JVM
jvm.start(system_cp=True, packages=True, max_heap_size="12288m")

# Load dataset
data_dir = f'datasets/{label}.arff'
dataset = converters.load_any_file(data_dir)
dataset.class_is_last()
logger.info(f'Dataset loaded...')

# Build the Classifier
classifier = Classifier(classname="weka.classifiers.functions.Logistic", options=[
    "-R", "2.280153483153162", "-M", "34",
    "-do-not-check-capabilities", "-num-decimal-places", "4"
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
plt.savefig(f'results/{label}-global.pdf', bbox_inches='tight', dpi=300)
plt.close()
logger.info(f"Saved SHAP plot for {label}")

# Log all stats from the explainer
logger.info("--------------- Global Statistics ---------------")
global_shap_df = pd.DataFrame(global_exp[:, :, 1].values, columns=feature_names)
logger.info("Shap Values: \n" + global_shap_df.abs().mean().sort_values(ascending=False).to_string())
logger.info("Shap Summary: \n" + str(global_shap_df.describe()))

# Load dataset for local explainability
test_dataset = converters.load_any_file(f'datasets/owasp-benchmark/{label}.arff')
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

# Create directory for saving plots and csv foles
output_dir = f'results/{label}-local/'
os.makedirs(output_dir, exist_ok=True)

csv_path = f'results/csv/{label}/'
os.makedirs(csv_path, exist_ok=True)

# Plot SHAP waterfall plots for local explainability
for i in range(0, len(test_dataset_np)):
    shap.plots.waterfall(local_exp[i,:dataset.num_attributes - 1, 1], show=False)
    plt.savefig(f'{output_dir}/{label}-local-instance-{i}.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved SHAP force plot for instance {i}")

    # Save local shap values to csv files
    instance_csv_path = f'{csv_path}/instance_{i}.csv'
    local_shap_df = pd.DataFrame([local_exp[i, :, 1].values], columns=feature_names)
    local_shap_df = local_shap_df.T.reset_index()
    local_shap_df.columns = ['Feature', 'SHAP Value']
    local_shap_df.sort_values(by='SHAP Value', ascending=False).to_csv(instance_csv_path, index=False)

# Log all stats from the explainer
logger.info("--------------- Local Statistics ---------------")
local_shap_df = pd.DataFrame(local_exp[:, :, 1].values, columns=feature_names)
logger.info("Shap Values: \n" + local_shap_df.abs().mean().sort_values(ascending=False).to_string())
logger.info("Shap Summary: \n" + str(local_shap_df.describe()))

# Stop JVM
jvm.stop()