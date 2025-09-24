import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from symptoms import SymptomMapper

logging.basicConfig(level=logging.DEBUG)

# Initialize symptom mapper
symptom_mapper = SymptomMapper(csv_path="./datasets/Symptom-severity.csv")

# Folder containing all datasets
dataset_folder = "./datasets"

# Collect all datasets
all_dfs = []
labels_set = set()

for file_name in os.listdir(dataset_folder):
    if not file_name.endswith(".csv"):
        continue
    file_path = os.path.join(dataset_folder, file_name)
    logging.debug(f"=== Loaded {file_name} ===")
    df = pd.read_csv(file_path)
    logging.debug(f"Shape: {df.shape}")
    all_dfs.append(df)
    
    # Detect labels
    label_cols = [col for col in df.columns if col.lower() in ['class','outcome','disease','heart_disease','fast_heart_rate']]
    if label_cols:
        logging.debug(f"Detected labels in {file_name}: {label_cols}")
        labels_set.update(label_cols)
    else:
        logging.warning(f"No labels detected in {file_name}, skipping label processing.")

# Combine all datasets (ignore index to avoid duplicates)
combined_df = pd.concat(all_dfs, ignore_index=True)
logging.debug("DEBUG: Final combined dataset ready")
logging.debug(f"Combined shape: {combined_df.shape}")

# Fill NaNs in feature columns with 0 (for symptoms/features)
feature_cols = [col for col in combined_df.columns if col not in labels_set]
combined_df[feature_cols] = combined_df[feature_cols].fillna(0)

# Extract features and labels
X = combined_df[feature_cols]
y = combined_df[list(labels_set)]

# Handle NaNs in labels by filling with a string
for col in y.columns:
    y[col] = y[col].fillna("Unknown")
    
    # Convert all entries to strings for LabelEncoder to avoid mixed type errors
    y[col] = y[col].astype(str)
    
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])
    logging.debug(f"Encoded label '{col}' with classes: {le.classes_}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Multi-output classifier with Random Forest
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

# Train
clf.fit(X_train, y_train)
logging.debug("Training complete!")

# Test accuracy
score = clf.score(X_test, y_test)
logging.debug(f"Test score: {score:.4f}")
