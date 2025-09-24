import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from symptoms import SymptomMapper  # your module

import logging
logging.basicConfig(level=logging.DEBUG)

# Initialize SymptomMapper
logging.debug("Initializing SymptomMapper...")
symptom_mapper = SymptomMapper(csv_path="./datasets/Symptom-severity.csv")
logging.debug(f"Loaded {len(symptom_mapper.symptom_to_weight)} symptoms from ./datasets/Symptom-severity.csv")

# Load datasets
dataset_folder = "./datasets"
all_dfs = []

for file_name in os.listdir(dataset_folder):
    if not file_name.endswith(".csv"):
        continue
    file_path = os.path.join(dataset_folder, file_name)
    logging.debug(f"=== Loaded {file_name} ===")
    df = pd.read_csv(file_path)
    logging.debug(f"Shape: {df.shape}")
    logging.debug(f"Columns (first 10): {df.columns[:10].tolist()}")

    # Detect labels (columns that are likely outputs)
    label_cols = [col for col in df.columns if col.lower() in ['class', 'outcome', 'disease', 'heart_disease', 'fast_heart_rate']]
    if label_cols:
        logging.debug(f"Detected labels in {file_name}: {label_cols}")
    else:
        logging.warning(f"WARNING: No labels detected in {file_name}, skipping this dataset.")
        continue

    # Map symptoms to weights if columns exist in symptom mapper
    symptom_cols = [col for col in df.columns if col in symptom_mapper.symptom_to_weight]
    if symptom_cols:
        logging.debug(f"Mapping symptoms in {file_name}: {symptom_cols[:5]}{'...' if len(symptom_cols) > 5 else ''}")

    # Add a dataset name column
    df['dataset_name'] = file_name
    all_dfs.append(df)

# Combine all datasets
full_df = pd.concat(all_dfs, ignore_index=True)
logging.debug("DEBUG: Final combined dataset ready")

# Separate features and labels
label_cols = ['Class', 'Outcome', 'disease', 'heart_disease', 'fast_heart_rate']
y = full_df[label_cols].copy()
X = full_df.drop(columns=label_cols + ['dataset_name'])

# Fill NaN in features with 0
X = X.fillna(0)

# Encode categorical labels safely
label_encoders = {}
for col in y.columns:
    # Convert all values to string first
    y[col] = y[col].astype(str)
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])
    label_encoders[col] = le
    logging.debug(f"Encoded label '{col}' with classes: {le.classes_}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)

# Evaluate
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
logging.debug(f"Training score: {train_score}")
logging.debug(f"Test score: {test_score}")

# Save model if needed
import joblib
joblib.dump(clf, "smart_health_model.pkl")
logging.debug("Model saved to smart_health_model.pkl")
