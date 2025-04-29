import logging
import os

import numpy as np
import shap
import weka.core.jvm as jvm
import weka.core.converters as converters
from matplotlib import pyplot as plt
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from Converters import inst_to_np
import weka.core.converters as con

label = 'source'

# Configure logger
logfile_path = os.getcwd() + f'/out/{label}.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logfile_path, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filemode='w', encoding='utf-8', level=logging.DEBUG, force=True)
print('logs saved at:' + logfile_path)

# Start JVM
jvm.start(system_cp=True, packages=True, max_heap_size="12288m")

# Load dataset
data_dir = f'Datasets_new/{label}.arff'
dataset = converters.load_any_file(data_dir)
dataset.class_is_last()
logger.info(f'Dataset loaded...')

# Train-Test split
# training_set, testing_set = dataset.train_test_split(70, Random(1))

# Build the Classifier
classifier = Classifier(classname="weka.classifiers.meta.ClassificationViaRegression", options=[
    "-W", "weka.classifiers.trees.RandomForest",
    "-do-not-check-capabilities", "--",
    "-P", "100", "-I", "100", "-num-slots", "1", "-do-not-check-capabilities",
    "-K", "0", "-M", "8.0", "-V", "0.001", "-S", "1"
])

logger.info('Classifier = ' + classifier.__str__())

# Train classifier
classifier.build_classifier(dataset)
logger.info('Model training completed successfully...')

# Evaluating the model
# evaluation = Evaluation(training_set)
# evaluation.test_model(classifier, testing_set)
# logger.info('Evaluation for ' + label + ': \n' + evaluation.summary())
# logger.info('F-Score: ' + evaluation.f_measure(0).__str__())
# logger.info('F-Score 1: ' + evaluation.f_measure(1).__str__())

feature_names = dataset.attribute_names()

def weka_predict(input_data):
    """ Convert numpy array to Instances and return the classifier's class distribution for each instance"""
    weka_data = con.ndarray_to_instances(input_data, 'swan', 'Att-!', feature_names)
    weka_data.class_is_last()
    predictions = []
    for inst in weka_data:
        pred = classifier.distribution_for_instance(inst)
        predictions.append(pred)
    return np.array(predictions)


# Convert dataset to NumPy array for Shap Explainer
dataset_np = inst_to_np(dataset)

# Compute SHAP values
explainer = shap.Explainer(weka_predict, dataset_np, feature_names=feature_names)
exp = explainer(dataset_np)

# Plot SHAP beeswarm plot
fig = plt.figure()
shap.plots.beeswarm(exp[:, :dataset.num_attributes - 1], show=False)
plt.xlabel('SHAP Value')
plt.savefig(f'results/{label}-global.pdf', bbox_inches='tight', dpi=300)
plt.close()
logger.info(f"Saved SHAP plot for {label}")

# Stop JVM
jvm.stop()
