import argparse
import logging
import os
import pandas as pd
import shap
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from skmultilearn.problem_transform import LabelPowerset

# Argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', required=True, help='Path to training ARFF file')
parser.add_argument('--test', required=True, help='Path to testing ARFF file')
parser.add_argument('--out', required=True, help='Output directory to store results')
args = parser.parse_args()

# Create output directories
output_path = f'{args.out}/results/dev-assist'
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'global'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'local/csv'), exist_ok=True)

# Configure logger
logfile_path = f'{args.out}/logs'
os.makedirs(logfile_path, exist_ok=True)
logfile_path = os.path.join(logfile_path, 'dev-assist.log')
logger = logging.getLogger("meka")
logging.basicConfig( filename=logfile_path, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                     filemode='w', encoding='utf-8', level=logging.INFO, force=True)
print('Logs saved at:', logfile_path)
logging.getLogger("shap").setLevel(logging.CRITICAL)
logging.getLogger("weka").setLevel(logging.CRITICAL)

def preprocessing(data_df):
    data_df = data_df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    data_df = data_df.drop(columns=['CWE601', 'CWE863', 'CWE306', 'CWE78', 'CWE862', 'authentication'])

    label_columns = data_df.columns[:5]
    feature_columns = data_df.columns[5:]
    data_df[label_columns] = data_df[label_columns].astype(int)

    categorical_boolean_columns = [col for col in data_df.columns if data_df[col].dtype == 'object']
    encoders = {}
    for col in categorical_boolean_columns:
        encoder = LabelEncoder()
        data_df[col] = encoder.fit_transform(data_df[col])
        encoders[col] = encoder

    logger.info('Preprocessing completed...')
    return data_df, label_columns, feature_columns

# Load the datasets
train_data, _ = arff.loadarff(args.train)
test_data, _ = arff.loadarff(args.test)
train_data_df = pd.DataFrame(train_data)
test_data_df = pd.DataFrame(test_data)

train_limit = train_data_df.shape[0]
data_df = pd.concat([train_data_df, test_data_df])
data_df, labels, features = preprocessing(data_df)

# Split into training and testing again
test_data_df = data_df.iloc[train_limit:]
data_df = data_df.iloc[:train_limit]
X_train = data_df[features]
y_train = data_df[labels]

# Train model
base_classifier = RandomForestClassifier(
    n_estimators=45,
    max_features=None,
    min_samples_split=2,
    min_samples_leaf=4,
    n_jobs=1,
    random_state=1,
    verbose=0,
    bootstrap=True
)
multi_label_model = LabelPowerset(base_classifier)
multi_label_model.fit(X_train, y_train)
logger.info('Model built successfully...')
logger.info("-----Powerset label mapping------")

# SHAP Global Explanation
logger.info('Resolving Shap...')
global_explainer = shap.Explainer(multi_label_model.classifier, X_train, seed=42)
global_exp = global_explainer(X_train)

for i in range(global_exp.shape[2]):
    fig = plt.figure()
    shap.plots.beeswarm(global_exp[:, :, i], show=False)
    plt.xlabel('SHAP Value')
    plt.savefig(os.path.join(output_path, f'global/LC-RF{i}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'Shap beeswarm plot exported for powerset: {i}...')

    global_shap_df = pd.DataFrame(global_exp[:, :, i].values, columns=features)
    logger.info(f"--------------- Global Statistics for Powerset {i} ---------------")
    logger.info("Shap Values: \n" + global_shap_df.mean().sort_values(ascending=False).to_string())

shap_values_aggregated_df = pd.DataFrame([global_exp[1,:,:].values.mean(axis=1)], columns=features)
logger.info("--------------- Aggregated Global Statistics ---------------")
logger.info(shap_values_aggregated_df.mean().sort_values(ascending=False).to_string())
shap_values_aggregated = global_exp.values.mean(axis=2)

fig = plt.figure()
shap.summary_plot(shap_values_aggregated, X_train, max_display=10, show=False)
plt.xlabel('SHAP Value')
plt.savefig(os.path.join(output_path, 'global/LC-RF_combined.pdf'), dpi=300, bbox_inches='tight')
plt.close()
logger.info(f'Shap beeswarm plot exported for aggregated powersets...')

# Testing
X_test = test_data_df[features]
y_test = test_data_df[labels]
y_pred = multi_label_model.predict(X_test)
logger.info('Macro f1: ' + str(f1_score(y_test, y_pred, average='macro')))
logger.info('Micro f1: ' + str(f1_score(y_test, y_pred, average='micro')))

# SHAP Local Explanation
local_explainer = shap.Explainer(multi_label_model.classifier, X_train, seed=42)
local_exp = local_explainer(X_test)

for i in range(X_test.shape[0]):
    csv_data = []
    for j in range(local_exp.shape[2]):
        fig = plt.figure()
        shap.plots.waterfall(local_exp[i, :, j], show=False)
        plt.savefig(os.path.join(output_path, f'local/LC-RF_Waterfall_{i}_class_{j}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'SHAP waterfall plot exported for test instance {i}, class {j}...')

        local_shap_df = pd.DataFrame([local_exp[i, :, j].values], columns=features)
        logger.info(f"--------------- Local Statistics for Instance {i} and label {j}---------------")
        logger.info("Shap Values: \n" + local_shap_df.abs().mean().sort_values(ascending=False).to_string())

        local_shap_df = local_shap_df.T.reset_index()
        local_shap_df.columns = ['Feature', 'SHAP Value']
        local_shap_df.insert(0, 'Label', j)
        csv_data.append(local_shap_df.sort_values(by='SHAP Value', ascending=False))

    local_shap_df = pd.DataFrame([local_exp[i, :].values.mean(axis=1)], columns=features)
    logger.info(f"--------------- Aggregated Local Statistics for Instance {i}---------------")
    logger.info("Shap Values: \n" + local_shap_df.mean().sort_values(ascending=False).to_string())

    final_df = pd.concat(csv_data, ignore_index=True)
    csv_path = os.path.join(output_path, f'local/csv/instance_{i}.csv')
    final_df.to_csv(csv_path, index=False)

logger.info(f'Finished Successfully...')
