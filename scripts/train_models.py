import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from data_preprocessing import preprocess_data
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

# Load and preprocess data
try:
    merged_data = pd.read_csv('../processed/merged_data.csv')
    if merged_data.empty:
        raise ValueError("Merged data is empty. Falling back to preprocessing.")
except (FileNotFoundError, ValueError):
    merged_data = preprocess_data()

# Split data
features = merged_data.drop(columns=['user_id', 'tgt'])
target = merged_data['tgt']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(rf_model, '../models/random_forest_model.pkl')

# Train XGBoost
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='auc')
xgb_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(xgb_model, '../models/xgboost_model.pkl')

# Train Logistic Regression
logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(logreg_model, '../models/logistic_regression_model.pkl')