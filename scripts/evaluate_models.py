import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess data
try:
    merged_data = pd.read_csv('../processed/merged_data.csv')
    if merged_data.empty:
        raise ValueError("Merged data is empty. Falling back to preprocessing.")
except (FileNotFoundError, ValueError):
    merged_data = preprocess_data()

# Check if merged_data is empty
if merged_data.empty:
    raise ValueError("Merged data is empty. Please check the preprocessing step.")

# Split data
features = merged_data.drop(columns=['user_id', 'tgt'])
target = merged_data['tgt']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Load models
rf_model = joblib.load('../models/random_forest_model.pkl')
xgb_model = joblib.load('../models/xgboost_model.pkl')
logreg_model = joblib.load('../models/logistic_regression_model.pkl')

# Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities)
    }

rf_results = evaluate_model(rf_model, X_test, y_test)
xgb_results = evaluate_model(xgb_model, X_test, y_test)
logreg_results = evaluate_model(logreg_model, X_test, y_test)

print("Random Forest Results:", rf_results)
print("XGBoost Results:", xgb_results)
print("Logistic Regression Results:", logreg_results)