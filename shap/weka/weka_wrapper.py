import argparse
import json
import os
import logging
import numpy as np
import pandas as pd
import shap
from weka.core import jvm, converters
from weka.classifiers import Classifier, Evaluation
import weka.core.converters as con
import matplotlib.pyplot as plt

from Converters import inst_to_np

np.random.seed(42)
config_path = 'shap/weka/config.json'

# Parser for approach
base_parser = argparse.ArgumentParser(description='', add_help=False)
base_parser.add_argument('--approach', required=True, choices=['swan', 'sscm'], help='Approach to use')
args, remaining_argv = base_parser.parse_known_args()

# Parser for output directory
parser = argparse.ArgumentParser(parents=[base_parser])
parser.add_argument('--out', required=True, help='Output directory to store results')

# Parser for label
if args.approach == 'swan':
    parser.add_argument('--label', required=False, help='Label to use for SWAN approach')

# Parse the label argument
args = parser.parse_args()
if args.approach == 'swan':
    label = args.label if args.label is not None else 'all'
else:
    label = args.approach

# Load classifier configurations
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
with open(config_path, 'r') as f:
    config = json.load(f)

# Create output directory
output_path = f'{args.out}/results/'
os.makedirs(output_path, exist_ok=True)

# Configure logger
logfile_path = f'{args.out}/logs'
os.makedirs(logfile_path, exist_ok=True)
logfile_path = os.path.join(logfile_path, f'{args.approach}-{label}.log')
logger = logging.getLogger(label)
logging.basicConfig(filename=logfile_path, format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='w', encoding='utf-8', level=logging.INFO, force=True)
print('logs saved at:' + logfile_path)
logging.getLogger("shap").setLevel(logging.CRITICAL)
logging.getLogger("weka").setLevel(logging.CRITICAL)

srms = []
if args.approach == 'swan':
    swan_labels = list(config[args.approach].keys())
    if label in swan_labels:
        srms = [label]
    else:
        if label == 'all':
            srms = swan_labels
        else:
            print(f"Label '{label}' not found in config file.")
            jvm.stop()
            exit(1)
if args.approach == 'sscm':
    srms =[args.approach]

# Start JVM
jvm.start(system_cp=True, packages=True, max_heap_size="12288m")
logger.info('JVM Started...')

for srm in srms:
    logger.info(f'eXRM WEKA-Wrapper -> approach:{args.approach}, label:{srm}')

    # Load train dataset
    dataset = converters.load_any_file(config[args.approach][srm]['train'])
    dataset.class_is_last()
    logger.info(f'Train Dataset loaded...')

    # Build the Classifier
    classifier = Classifier(classname=config[args.approach][srm]['classifier'], options=config[args.approach][srm]['options'])
    logger.info('Classifier = ' + classifier.__str__())

    # Train classifier
    classifier.build_classifier(dataset)
    logger.info('Model training completed successfully...')

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
    dataset_np = shap.sample(dataset_np, 15)

    # Compute SHAP values
    global_explainer = shap.Explainer(weka_predict, dataset_np, feature_names=feature_names, seed=42)
    global_exp = global_explainer(dataset_np)
    fig = plt.figure()

    logger.info("--------------- SHAP Global ---------------")
    if args.approach == 'sscm':
        for i in range(0, 3):
            # Generate and save beeswarm plot
            shap.plots.beeswarm(global_exp[:, :dataset.num_attributes - 1, i], show=False)
            plt.xlabel('SHAP Value')
            plt.savefig(f'{output_path}/{srm}-global-tag_{i}.pdf', bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Saved SHAP plot for {srm} class {i}")

            # Log SHAP values for all features
            logger.info(f"--------------- Label {i}---------------")
            global_shap_df = pd.DataFrame(global_exp[:, :, i].values, columns=feature_names)
            logger.info("Shap Values: \n" + global_shap_df.abs().mean().sort_values(ascending=False).to_string())
    else:
        if config[args.approach][srm]['regressor']:
            global_shap = global_exp[:, :dataset.num_attributes - 1]
            global_shap_val = global_exp.values
        else:
            global_shap = global_exp[:, :dataset.num_attributes - 1, 1]
            global_shap_val = global_exp[:, :, 1].values

        # Generate and save beeswarm plot
        shap.plots.beeswarm(global_shap, show=False)
        plt.xlabel('SHAP Value')
        plt.savefig(f'{output_path}/{srm}-global.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved SHAP plot for {srm}")

        # Log SHAP values for all features
        global_shap_df = pd.DataFrame(global_shap_val, columns=feature_names)
        logger.info("Shap Values: \n" + global_shap_df.abs().mean().sort_values(ascending=False).to_string())

    # Load test dataset
    test_dataset = converters.load_any_file(config[args.approach][srm]['test'])
    test_dataset.class_is_last()
    logger.info(f'Test Dataset loaded...')

    # Evaluate the classifier on the test dataset
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

    # Output directory for plots and CSV files
    os.makedirs(os.path.join(output_path, f'local/{srm}'), exist_ok=True)
    os.makedirs(os.path.join(output_path, f'local/csv/{srm}'), exist_ok=True)

    logger.info("--------------- SHAP Local ---------------")

    for i in range(0, len(test_dataset_np)):
        if args.approach == 'sscm':
            local_shap_values = []
            for tag in range(3):
                # Generate and save waterfall plots
                shap.plots.waterfall(local_exp[i, :dataset.num_attributes - 1, tag], show=False)
                plt.savefig(f'{output_path}/local/{srm}/{srm}-instance-{i}-tag_{tag}.pdf', bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"Saved SHAP waterfall plot for instance {i} and tag {tag}")

                # Save local shap values to csv files
                df = pd.DataFrame([local_exp[i, :, tag].values], columns=feature_names)
                df = df.T.reset_index()
                df.columns = ['Feature', 'SHAP Value']
                df['Tag'] = f'tag_{tag}'
                local_shap_values.append(df)

            instance_csv_path = f'{output_path}/local/csv/{srm}/instance_{i}.csv'
            local_shap_df = pd.concat(local_shap_values, ignore_index=True)
            local_shap_df.to_csv(instance_csv_path, index=False)
            logger.info(f"Local SHAP values saved for instance {i} at {instance_csv_path}")

        else:
            # Generate and save waterfall plots
            if config[args.approach][srm]['regressor']:
                local_shap = local_exp[i, :dataset.num_attributes - 1]
            else:
                local_shap = local_exp[i, :dataset.num_attributes - 1,1]
            shap.plots.waterfall(local_shap, show=False)
            plt.savefig(f'{output_path}/local/{srm}/local-instance-{i}.pdf', bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Saved SHAP waterfall plot for instance {i}")

            # Save local shap values to csv files
            instance_csv_path = f'{output_path}/local/csv/{srm}/instance_{i}.csv'
            local_shap_df = pd.DataFrame([local_shap.values], columns=feature_names[:len(feature_names)-1])
            local_shap_df = local_shap_df.T.reset_index()
            local_shap_df.columns = ['Feature', 'SHAP Value']
            local_shap_df.sort_values(by='SHAP Value', ascending=False).to_csv(instance_csv_path, index=False)
            logger.info(f"Local SHAP values saved for instance {i} at {instance_csv_path}")

jvm.stop()

