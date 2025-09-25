import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

logging.basicConfig(level=logging.DEBUG)

# ------------------------
# Load dataset
# ------------------------
dataset_path = r"E:\Projects\Smart Health Predictor\Smart-Health-Predictor\datasets\master_dataset.csv"
logging.debug(f"Dataset found at: {dataset_path}")

df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# ------------------------
# Identify label and features
# ------------------------
label_column = 'prognosis'  # target column
X = df.drop(label_column, axis=1)
y = df[label_column]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')
logging.debug(f"Label encoder classes saved: {le.classes_}")

# ------------------------
# Handle missing values
# ------------------------
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
logging.debug("Missing values handled with median imputation")

# ------------------------
# Feature selection
# ------------------------
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_imputed, y_encoded)
logging.debug("Selected top 100 features")

# ------------------------
# SMOTE to handle imbalance
# ------------------------
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y_encoded)
logging.debug(f"Resampled dataset shape: {X_resampled.shape}")

# ------------------------
# Train model
# ------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_resampled, y_resampled)
logging.debug("Model training completed")

# ------------------------
# Save the trained model
# ------------------------
joblib.dump(model, 'multidisease_model.pkl')
logging.debug("Trained model saved as multidisease_model.pkl")
