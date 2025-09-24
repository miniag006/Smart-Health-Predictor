import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from symptoms import SymptomMapper

logging.basicConfig(level=logging.DEBUG)

# Initialize SymptomMapper
logging.debug("Initializing SymptomMapper...")
symptom_mapper = SymptomMapper()  # Using default initialization
logging.debug(f"Loaded {len(symptom_mapper.symptoms)} symptoms from ./datasets/Symptom-severity.csv")

# Scan datasets folder
datasets_folder = "./datasets"
combined_df = pd.DataFrame()
labels_set = set()

for file_name in os.listdir(datasets_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(datasets_folder, file_name)
        logging.debug(f"=== Loaded {file_name} ===")
        df = pd.read_csv(file_path)
        logging.debug(f"Shape: {df.shape}")

        # Detect label columns
        label_cols = [col for col in df.columns if col.lower() in ['class', 'outcome', 'disease', 'heart_disease', 'fast_heart_rate']]
        if not label_cols:
            logging.warning(f"No labels detected in {file_name}, skipping label processing.")
        else:
            logging.debug(f"Detected labels in {file_name}: {label_cols}")
            labels_set.update(label_cols)

        # Map symptom columns if applicable
        symptom_cols = [col for col in df.columns if col in symptom_mapper.symptoms]
        if symptom_cols:
            logging.debug(f"Mapping symptoms in {file_name}: {symptom_cols}...")
            for col in symptom_cols:
                df[col] = df[col].apply(lambda x: symptom_mapper.map_symptom(x))
        
        combined_df = pd.concat([combined_df, df], ignore_index=True)

logging.debug("DEBUG: Final combined dataset ready")
logging.debug(f"Combined shape: {combined_df.shape}")

# Fill NaNs in feature columns
feature_cols = [col for col in combined_df.columns if col not in labels_set]
combined_df[feature_cols] = combined_df[feature_cols].fillna("Unknown")

# Convert all non-numeric feature columns into numeric
for col in feature_cols:
    if combined_df[col].dtype == 'object' or isinstance(combined_df[col].iloc[0], str):
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))

# Extract features and labels
X = combined_df[feature_cols]
y = combined_df[list(labels_set)]

# Encode label columns
for col in y.columns:
    y[col] = y[col].fillna("Unknown")
    y[col] = y[col].astype(str)
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])
    logging.debug(f"Encoded label '{col}' with classes: {le.classes_}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifier
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)
logging.debug("Training complete.")

# Optional: save the trained model
import joblib
joblib.dump(clf, "smart_health_model.pkl")
logging.debug("Model saved to smart_health_model.pkl")
