import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from symptoms import SymptomMapper

logging.basicConfig(level=logging.DEBUG)

# Initialize SymptomMapper
symptom_mapper = SymptomMapper()
logging.debug(f"Loaded {len(symptom_mapper.symptom_to_weight)} symptoms")

# Path to datasets folder
datasets_folder = "./datasets"

# List to hold all datasets
dfs = []

# Scan datasets folder
for file_name in os.listdir(datasets_folder):
    if file_name.endswith(".csv"):
        path = os.path.join(datasets_folder, file_name)
        df = pd.read_csv(path)
        logging.debug(f"=== Loaded {file_name} ===")
        logging.debug(f"Shape: {df.shape}")
        
        # Add dataset_name for tracking
        df['dataset_name'] = file_name
        
        # Detect label columns: assume last column is the label if it is not numeric
        label_cols = [col for col in df.columns if col.lower() in ['class','outcome','disease','heart_disease','fast_heart_rate']]
        if label_cols:
            logging.debug(f"Detected labels in {file_name}: {label_cols}")
        else:
            logging.warning(f"No labels detected in {file_name}, skipping label processing.")
        
        dfs.append(df)

# Combine all datasets
combined_df = pd.concat(dfs, ignore_index=True)
logging.debug(f"DEBUG: Final combined dataset ready")
logging.debug(f"Combined shape: {combined_df.shape}")

# Define features and labels
label_cols = ['Class', 'Outcome', 'disease', 'heart_disease', 'fast_heart_rate']
X = combined_df.drop(columns=[col for col in label_cols if col in combined_df.columns])
y = combined_df[[col for col in label_cols if col in combined_df.columns]]

# Convert symptom strings to numeric using SymptomMapper
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].apply(symptom_mapper.get_weight)

# Encode labels
for col in y.columns:
    y[col] = y[col].fillna("Unknown").astype(str)
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])
    logging.debug(f"Encoded label '{col}' with classes: {le.classes_}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize multi-output classifier
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)
logging.debug("Model training completed!")

# Evaluate model
score = clf.score(X_test, y_test)
logging.debug(f"Model accuracy: {score:.4f}")

import joblib

# Save the trained model
joblib.dump(clf, "multi_disease_model.pkl")
print("Model saved successfully!")

