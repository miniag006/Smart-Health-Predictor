import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = "datasets/master_dataset.csv"
df = pd.read_csv(dataset_path)
logging.debug(f"Dataset found at: {dataset_path}")
logging.debug(f"Master dataset shape: {df.shape}")

# Define label column
label_col = 'disease'
X = df.drop(label_col, axis=1)
y = df[label_col]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
logging.debug(f"Label encoder classes saved: {le.classes_}")
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Handle missing values
if numeric_cols:
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

logging.debug("Missing values handled for numeric and categorical columns")

# One-hot encode categorical columns
if categorical_cols:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = pd.DataFrame(ohe.fit_transform(X[categorical_cols]))
    X_encoded.index = X.index  # align indices
    X = X.drop(categorical_cols, axis=1)
    X = pd.concat([X, X_encoded], axis=1)

# Feature selection: top 100 features
selector = SelectKBest(score_func=f_classif, k=min(100, X.shape[1]))
X_selected = selector.fit_transform(X, y_encoded)
logging.debug("Selected top 100 features")

# Apply SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y_encoded)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_resampled, y_resampled)

# Save the trained model
with open('multidisease_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

logging.debug("Model training completed and saved successfully")
