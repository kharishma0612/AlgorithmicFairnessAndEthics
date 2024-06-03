import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

data = data.dropna()
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
data = pd.get_dummies(data, drop_first=True)

X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sensitive_feature_train = X_train[:, X.columns.get_loc('sex_Male')]
sensitive_feature_test = X_test[:, X.columns.get_loc('sex_Male')]

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

metric_frame_baseline = MetricFrame(
    metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
    y_true=y_test,
    y_pred=y_pred_baseline,
    sensitive_features=sensitive_feature_test
)

selection_rate_diff_baseline = metric_frame_baseline.difference(method='between_groups')['selection_rate']


selection_rate_diff_mitigated = metric_frame_mitigated.difference(method='between_groups')['selection_rate']

postprocess = ThresholdOptimizer(estimator=baseline_model, constraints="demographic_parity", predict_method='predict_proba')
postprocess.fit(X_train, y_train, sensitive_features=sensitive_feature_train)
y_pred_postprocessed = postprocess.predict(X_test, sensitive_features=sensitive_feature_test)

metric_frame_postprocessed = MetricFrame(
    metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
    y_true=y_test,
    y_pred=y_pred_postprocessed,
    sensitive_features=sensitive_feature_test
)

selection_rate_diff_postprocessed = metric_frame_postprocessed.difference(method='between_groups')['selection_rate']

unique_values = np.unique(sensitive_feature_test)
female_value = unique_values[0]
male_value = unique_values[1]

print("=== Fairness Evaluation Results ===")

print("\nBaseline Model:")
print("Selection Rate per Group:")
print("Female Selection Rate: {:.1%}".format(metric_frame_baseline.by_group['selection_rate'][female_value]))
print("Male Selection Rate: {:.1%}".format(metric_frame_baseline.by_group['selection_rate'][male_value]))
print("Demographic Parity Difference (Selection Rate Difference): {:.1%}".format(selection_rate_diff_baseline))


print("\nPost-processed Model:")
print("Selection Rate per Group:")
print("Female Selection Rate: {:.1%}".format(metric_frame_postprocessed.by_group['selection_rate'][female_value]))
print("Male Selection Rate: {:.1%}".format(metric_frame_postprocessed.by_group['selection_rate'][male_value]))
print("Demographic Parity Difference (Selection Rate Difference): {:.1%}".format(selection_rate_diff_postprocessed))
