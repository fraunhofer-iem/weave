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
logfile_path = os.getcwd() + '/logs.log'
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
train_file_path = "Datasets/train-meka.arff"
test_file_path = "Datasets/test-meka.arff"
train_data, train_meta = arff.loadarff(train_file_path)
test_data, test_meta = arff.loadarff(test_file_path)
train_data_df = pd.DataFrame(train_data)
test_data_df = pd.DataFrame(test_data)
logger.info('Datasets loaded successfully...')

train_data_df, train_labels, train_features = preprocessing(train_data_df)
test_data_df, test_labels, test_features = preprocessing(test_data_df)

# Train-Test split
X_train = train_data_df[train_features]
y_train = train_data_df[train_labels]
X_test = test_data_df[test_features]
y_test = test_data_df[test_labels]

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

# Evaluate the model
y_pred = multi_label_model.predict(X_test)
logger.info('Macro f1:' + str(f1_score(y_test, y_pred, average='macro')))
logger.info('Micro f1:' + str(f1_score(y_test, y_pred, average='micro')))

# Plot SHAP beeswarm plot
logger.info('Resolving Shap...')
explainer = shap.Explainer(multi_label_model.classifier, X_train)
shap_values = explainer(X_test)

for i in range(0, shap_values.shape[2]):
    fig = plt.figure()
    shap.plots.beeswarm(shap_values[:, :, i], show=False)
    plt.savefig(f"out/LC-RF{i}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'Shap beeswarm plot exported for powerset: {i}...')



# Aggregating shap values
shap_values_aggregated = shap_values.values.mean(axis=2)

# Plot the single aggregated beeswarm plot
fig = plt.figure()
shap.summary_plot(shap_values_aggregated, X_test, show=False)
plt.savefig("out/LC-RF_combined.pdf", dpi=300, bbox_inches='tight')
plt.close()

logger.info(f'Shap beeswarm plot exported for aggregated powersets...')
logger.info(f'Finished Successfully...')
