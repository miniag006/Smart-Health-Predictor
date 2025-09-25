import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = "datasets/master_dataset.csv"
df = pd.read_csv(dataset_path)
logging.debug(f"Dataset found at: {dataset_path}")
logging.debug(f"Master dataset shape: {df.shape}")

# Identify label column
label_col = 'prognosis'  # replace with your actual label column if different
X = df.drop(label_col, axis=1)
y = df[label_col]

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
logging.debug(f"Label encoder classes saved: {label_encoder.classes_}")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

cat_imputer = SimpleImputer(strategy='most_frequent')
X_categorical = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

# One-hot encode categorical features
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_categorical_encoded = pd.DataFrame(ohe.fit_transform(X_categorical), columns=ohe.get_feature_names_out(categorical_cols))

# Combine numeric and encoded categorical features
X_final = pd.concat([X_numeric, X_categorical_encoded], axis=1)
logging.debug(f"Final feature matrix shape: {X_final.shape}")

# Feature selection (optional)
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_final, y_encoded)
logging.debug("Selected top 100 features")

# Apply SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y_encoded)
logging.debug(f"Resampled dataset shape: {X_resampled.shape}")

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
logging.debug("Model training completed")

# Save model
with open("multidisease_model.pkl", "wb") as f:
    pickle.dump(model, f)
logging.debug("Model saved as multidisease_model.pkl")
