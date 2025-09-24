# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib  # for saving/loading models
from symptoms import SymptomMapper

# --- CONFIG ---
DATA_DIR = "./datasets"
SYMPTOM_FILE = os.path.join(DATA_DIR, "Symptom-severity.csv")
MODEL_FILE = "trained_model.pkl"
FEATURES_FILE = "feature_columns.pkl"

# --- INITIALIZE SYMPTOM MAPPER ---
print("DEBUG: Initializing SymptomMapper...")
symptom_mapper = SymptomMapper(SYMPTOM_FILE)
print(f"DEBUG: Loaded {len(symptom_mapper.list_symptoms())} symptoms from {SYMPTOM_FILE}")

# --- HELPER FUNCTION TO DETECT LABELS ---
def detect_label_columns(df):
    possible_labels = ["Class", "Outcome", "disease", "heart_disease", "fast_heart_rate"]
    labels = [col for col in df.columns if col in possible_labels]
    return labels

# --- LOAD AND PROCESS DATASETS ---
all_data = []
print(f"\nDEBUG: Scanning datasets folder: {DATA_DIR}")
for file_name in os.listdir(DATA_DIR):
    if not file_name.endswith(".csv") or file_name == "Symptom-severity.csv":
        continue

    path = os.path.join(DATA_DIR, file_name)
    try:
        df = pd.read_csv(path)
        print(f"\n=== Loaded {file_name} ===")
        print(f"Shape: {df.shape}")
        print(f"Columns (first 10): {list(df.columns)[:10]}")

        # Map symptoms to weights
        symptom_cols = [col for col in df.columns if col in symptom_mapper.list_symptoms()]
        if symptom_cols:
            print(f"Mapping symptoms in {file_name}: {symptom_cols[:5]}...")
            for col in symptom_cols:
                df[col] = df[col].map(symptom_mapper.get_weight).fillna(0)

        # Detect labels
        labels = detect_label_columns(df)
        if not labels:
            print(f"WARNING: No labels detected in {file_name}, skipping this dataset.")
            continue

        # Drop rows where all label columns are NaN
        df = df.dropna(subset=labels, how="all")
        if df.empty:
            print(f"WARNING: After cleaning, no valid label data in {file_name}, skipping.")
            continue

        print(f"Detected labels in {file_name}: {labels}")
        df['dataset_name'] = file_name
        all_data.append((df, labels))

    except Exception as e:
        print(f"ERROR reading {file_name}: {e}")

if not all_data:
    raise ValueError("No datasets loaded. Please check your files and paths!")

# --- COMBINE DATA ---
feature_frames = []
label_frames = []
for df, labels in all_data:
    feature_cols = df.select_dtypes(include='number').columns.tolist()
    feature_cols = [c for c in feature_cols if c not in labels]
    feature_frames.append(df[feature_cols])
    label_frames.append(df[labels])

X = pd.concat(feature_frames, ignore_index=True)
y = pd.concat(label_frames, ignore_index=True)

# --- CLEAN LABELS ---
# Remove columns that are completely empty
y = y.dropna(axis=1, how="all")
# Fill remaining NaNs with a placeholder
y = y.fillna("Unknown")

# Encode categorical labels
label_encoders = {}
for col in y.columns:
    if y[col].dtype == object:
        le = LabelEncoder()
        y[col] = le.fit_transform(y[col])
        label_encoders[col] = le
        print(f"DEBUG: Encoded label '{col}' with classes: {le.classes_}")

print("\nDEBUG: Final combined dataset ready")
print(f"Feature matrix shape: {X.shape}")
print(f"Label matrix shape: {y.shape}")
print(f"All labels being trained on: {y.columns.tolist()}")

# --- SPLIT AND TRAIN MULTI-LABEL MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# --- RESULTS ---
for i, label in enumerate(y.columns):
    print(f"\n=== Results for {label} ===")
    print(f"Accuracy: {accuracy_score(y_test.iloc[:, i], y_pred[:, i]):.4f}")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# --- SAVE MODEL AND FEATURES ---
joblib.dump(clf, MODEL_FILE)
joblib.dump(X.columns.tolist(), FEATURES_FILE)
joblib.dump(label_encoders, "label_encoders.pkl")
print(f"\nDEBUG: Training complete. Model saved as {MODEL_FILE}")
print(f"DEBUG: Feature columns saved as {FEATURES_FILE}")
print(f"DEBUG: Label encoders saved as label_encoders.pkl")

# --- FINAL SUMMARY ---
print("\n=== TRAINING SUMMARY ===")
print(f"Datasets used: {len(all_data)}")
print("Included datasets:", [df['dataset_name'].iloc[0] for df, _ in all_data])
print(f"Total samples: {len(X)}")
print(f"Total features: {X.shape[1]}")
print(f"Trained labels: {y.columns.tolist()}")
print("========================")
