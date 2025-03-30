import pandas as pd
import joblib
from data_preprocessing import preprocess_test_data
import os


# Load test data
test_data_path = '../processed/test_data.csv'
try:
    test_data = pd.read_csv(test_data_path)
    if test_data.empty:
        raise ValueError("Test data is empty. Please check the file.")
except (FileNotFoundError, ValueError):
    raise FileNotFoundError(f"Test data not found or is empty at {test_data_path}")

# Preprocess test data if necessary
# Assuming preprocess_data can handle test data preprocessing
test_features = test_data.drop(columns=['user_id'])

# Load models
models = {
    "random_forest": '../models/random_forest_model.pkl',
    "xgboost": '../models/xgboost_model.pkl',
    "logistic_regression": '../models/logistic_regression_model.pkl'
}

predictions = {}

# Add user_id column to prediction output
for model_name, model_path in models.items():
    model = joblib.load(model_path)
    probabilities = model.predict_proba(test_features)[:, 1]
    predictions[model_name] = probabilities
    output_path = f"../predictions/{model_name}_test_probabilities.csv"
    # Include user_id in the output
    output_df = pd.DataFrame({
        'user_id': test_data['user_id'],
        'pred_proba': probabilities
    })
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")