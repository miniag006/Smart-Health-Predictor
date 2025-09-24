# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from symptoms import SymptomMapper

# --- CONFIG ---
DATA_DIR = './datasets'
MODEL_PATH = './trained_model.pkl'

# --- HELPER FUNCTIONS ---
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_data(df, symptom_mapper=None):
    """Converts symptom names to weights if symptom_mapper is provided."""
    X = df.drop(columns=[col for col in df.columns if 'prognosis' in col.lower() or 'outcome' in col.lower() or 'class' in col.lower()], errors='ignore')
    y_col = [col for col in df.columns if 'prognosis' in col.lower() or 'outcome' in col.lower() or 'class' in col.lower()]
    y = df[y_col[0]] if y_col else None

    if symptom_mapper:
        for col in X.columns:
            if col in symptom_mapper.symptom_weights:
                X[col] = X[col] * symptom_mapper.symptom_weights[col]

    return X, y

# --- LOAD DATASETS ---
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
dfs = [load_dataset(f) for f in all_files]

# Concatenate all datasets (only those with labels)
labeled_dfs = []
for df in dfs:
    label_cols = [col for col in df.columns if 'prognosis' in col.lower() or 'outcome' in col.lower() or 'class' in col.lower()]
    if label_cols:
        labeled_dfs.append(df)

full_df = pd.concat(labeled_dfs, ignore_index=True)

# --- PREPARE FEATURES AND LABELS ---
symptom_mapper = SymptomMapper('./datasets/Symptom-severity.csv')
X, y = prepare_data(full_df, symptom_mapper=symptom_mapper)

# Fill missing values if any
X = X.fillna(0)
y = y.fillna(method='ffill')  # or choose another strategy

# --- SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAIN MODEL ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {acc*100:.2f}%')

# --- SAVE MODEL ---
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(clf, f)

print(f'Model saved to {MODEL_PATH}')
