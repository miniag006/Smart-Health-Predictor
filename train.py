import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.DEBUG)

# --- Dataset path ---
dataset_path = r"E:\Projects\Smart Health Predictor\Smart-Health-Predictor\datasets\master_dataset.csv"

# Load dataset
logging.debug(f"Dataset found at: {dataset_path}")
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# --- Label encoding ---
label_column = 'prognosis'
y = df[label_column]
X = df.drop(label_column, axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")
logging.debug(f"Label encoder classes saved: {label_encoder.classes_}")

# --- Handle missing values ---
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

# Impute numeric with median
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# Impute categorical with mode
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

logging.debug("Missing values handled for numeric and categorical columns")

# Optional: convert categorical columns to numeric (if needed)
X = pd.get_dummies(X, drop_first=True)

# --- Feature selection ---
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y_encoded)
logging.debug("Selected top 100 features")

# --- SMOTE oversampling ---
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y_encoded)
logging.debug(f"Resampled dataset shape: {X_resampled.shape}")

# --- Train model ---
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_resampled, y_resampled)
logging.debug("Model training completed")

# --- Save trained model ---
joblib.dump(model, "multidisease_model.pkl")
logging.debug("Trained model saved as multidisease_model.pkl")
