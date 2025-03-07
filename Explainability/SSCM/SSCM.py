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

label = 'sscm'

# Configure logger
logfile_path = os.getcwd() + f'/out/{label}.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logfile_path, format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filemode='w', encoding='utf-8', level=logging.DEBUG, force=True)
print('logs saved at:' + logfile_path)

# Start JVM
jvm.start(system_cp=True, packages=True, max_heap_size="12288m")

# Load dataset
data_dir = f'Datasets_new/{label}.arff'
dataset = converters.load_any_file(data_dir)
dataset.class_is_last()
logger.info(f'Dataset loaded...')

# Train-Test split
dataset.stratify(10)
training_set = dataset.train_cv(10, 0, Random(1))
testing_set = dataset.test_cv(10, 0)

# Build the Classifier

classifier = Classifier(classname="weka.classifiers.lazy.IBk", options=[
    "-K", "7", "-W", "0", "-X", "-A",
    "weka.core.neighboursearch.LinearNNSearch", "-do-not-check-capabilities"
])


logger.info('Classifier = ' + classifier.__str__())

# Train classifier
classifier.build_classifier(training_set)
logger.info('Model training completed successfully...')

# Evaluating the model
evaluation = Evaluation(training_set)
evaluation.test_model(classifier, testing_set)
logger.info('Evaluation for ' + label + ': \n' + evaluation.summary())
logger.info('F-Score: ' + evaluation.f_measure(0).__str__())
logger.info('F-Score 1: ' + evaluation.f_measure(1).__str__())


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


# Convert dataset to NumPy array
X_train = inst_to_np(training_set)
X_test = inst_to_np(testing_set)

# Compute SHAP values
explainer = shap.Explainer(weka_predict, X_train, feature_names=feature_names)
exp = explainer(X_test)


# Plot SHAP beeswarm plot
fig = plt.figure()
shap.plots.beeswarm(exp[:, :training_set.num_attributes - 1, 0], show=False)
plt.xlabel('SHAP Value')
plt.savefig(f'out/{label}-global.pdf', bbox_inches='tight', dpi=300)
plt.close()
logger.info(f"Saved SHAP plot for {label}")


# Stop JVM
jvm.stop()
