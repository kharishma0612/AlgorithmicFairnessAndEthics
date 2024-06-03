!pip install fairlearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

# Drop rows with missing values
data = data.dropna()

# Convert target variable to binary
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop('income', axis=1)
y = data['income']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Extract sensitive features (gender in this case)
sensitive_feature_train = X_train[:, X.columns.get_loc('sex_Male')]
sensitive_feature_test = X_test[:, X.columns.get_loc('sex_Male')]

# Train a baseline logistic regression model
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

# Evaluate the baseline model using fairness metrics
metric_frame_baseline = MetricFrame(
    metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
    y_true=y_test,
    y_pred=y_pred_baseline,
    sensitive_features=sensitive_feature_test
)
selection_rate_diff_baseline = metric_frame_baseline.difference(method='between_groups')['selection_rate']

# Evaluate accuracy of the baseline model
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)

# Define the fairness constraint
constraint = DemographicParity()

# Apply Exponentiated Gradient method for fairness mitigation
mitigator = ExponentiatedGradient(estimator=LogisticRegression(max_iter=1000), constraints=constraint)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_feature_train)
y_pred_mitigated = mitigator.predict(X_test)

# Evaluate the mitigated model using fairness metrics
metric_frame_mitigated = MetricFrame(
    metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
    y_true=y_test,
    y_pred=y_pred_mitigated,
    sensitive_features=sensitive_feature_test
)
selection_rate_diff_mitigated = metric_frame_mitigated.difference(method='between_groups')['selection_rate']

# Evaluate accuracy of the mitigated model
accuracy_mitigated = accuracy_score(y_test, y_pred_mitigated)

# Print the results
unique_values = np.unique(sensitive_feature_test)
female_value = unique_values[0]
male_value = unique_values[1]

print("=== Fairness Evaluation Results ===")

# Baseline Model results
print("\nBaseline Model:")
print("Selection Rate per Group:")
print("Female Selection Rate: {:.1%}".format(metric_frame_baseline.by_group['selection_rate'][female_value]))
print("Male Selection Rate: {:.1%}".format(metric_frame_baseline.by_group['selection_rate'][male_value]))
print("Demographic Parity Difference (Selection Rate Difference): {:.1%}".format(selection_rate_diff_baseline))
print("Accuracy: {:.1%}".format(accuracy_baseline))

# Mitigated Model results
print("\nMitigated Model (Exponentiated Gradient applied):")
print("Selection Rate per Group:")
print("Female Selection Rate: {:.1%}".format(metric_frame_mitigated.by_group['selection_rate'][female_value]))
print("Male Selection Rate: {:.1%}".format(metric_frame_mitigated.by_group['selection_rate'][male_value]))
print("Demographic Parity Difference (Selection Rate Difference): {:.1%}".format(selection_rate_diff_mitigated))
print("Accuracy: {:.1%}".format(accuracy_mitigated))
