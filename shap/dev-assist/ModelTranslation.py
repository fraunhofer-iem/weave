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

# Configure logger
logfile_path = os.getcwd() + '/out/logs.log'
logging.basicConfig(filename=logfile_path, filemode='w', encoding='utf-8', level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
print('logs saved at:' + logfile_path)


def preprocessing(data_df):
    # pre-processing the data
    data_df = data_df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    data_df = data_df.drop(columns=['CWE601', 'CWE863', 'CWE306', 'CWE78', 'CWE862', 'authentication'])

    # Encoding non-numeric data
    label_columns = data_df.columns[:5]
    feature_columns = data_df.columns[5:]
    data_df[label_columns] = data_df[label_columns].astype(int)
    categorical_boolean_columns = [col for col in data_df.columns if data_df[col].dtype == 'object']
    for col in categorical_boolean_columns:
        encoder = LabelEncoder()
        data_df[col] = encoder.fit_transform(data_df[col])

    logger.info('Preprocessing completed...')
    return data_df, label_columns, feature_columns


# Load the datasets
dataset_file = "Datasets/dataset.arff"
data, meta_data = arff.loadarff(dataset_file)
data_df = pd.DataFrame(data)
logger.info('Datasets loaded successfully...')

data_df, labels, features = preprocessing(data_df)

# Train-Test split
X_train = data_df[features]
y_train = data_df[labels]

# Train the model
base_classifier = RandomForestClassifier(
    n_estimators=45,  # -I 45 (Number of trees)
    max_features=None,  # -K 0 (Use all features)
    min_samples_split=2,  # -M 4.0 (min instances per leaf)
    min_samples_leaf=4,  # -M 4.0 (Minimum samples per leaf)
    n_jobs=1,  # -num-slots 1 (Number of parallel jobs)
    random_state=1,  # -S 1 (Random seed)
    verbose=0,  # -V 1.0E-5
    bootstrap=True  # -B (Use bagging)
)
multi_label_model = LabelPowerset(base_classifier)
multi_label_model.fit(X_train, y_train)
logger.info('Model built successfully...')

# Plot SHAP beeswarm plot
logger.info('Resolving Shap...')
explainer = shap.Explainer(multi_label_model.classifier, X_train)
shap_values = explainer(X_train)

for i in range(0, shap_values.shape[2]):
    fig = plt.figure()
    shap.plots.beeswarm(shap_values[:, :, i], show=False)
    plt.savefig(f"results/LC-RF{i}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'Shap beeswarm plot exported for powerset: {i}...')



# Aggregating shap values
shap_values_aggregated = shap_values.values.mean(axis=2)

# Plot the single aggregated beeswarm plot
fig = plt.figure()
shap.summary_plot(shap_values_aggregated, X_train, show=False)
plt.xlabel('SHAP Value')
plt.savefig("results/LC-RF_combined.pdf", dpi=300, bbox_inches='tight')
plt.close()

logger.info(f'Shap beeswarm plot exported for aggregated powersets...')
logger.info(f'Finished Successfully...')
