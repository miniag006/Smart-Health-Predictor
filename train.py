import pandas as pd
import logging
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

logging.basicConfig(level=logging.DEBUG)

# Load master dataset
master_path = "./datasets/master_dataset.csv"
df = pd.read_csv(master_path)
logging.debug(f"Master dataset shape: {df.shape}")

# Separate features and target
target_col = "disease"
y = df[target_col]
X = df.drop(columns=[target_col])

# Identify numerical columns (all non-binary/symptom columns)
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
binary_cols = [col for col in X.columns if col not in numerical_cols]

logging.debug(f"Numerical columns: {numerical_cols}")
logging.debug(f"Binary/Symptom columns: {binary_cols}")

# Impute missing values
# Numerical: median
num_imputer = SimpleImputer(strategy="median")
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

# Binary/Symptom: fill NaN with 0
X[binary_cols] = X[binary_cols].fillna(0)

# Scale numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
logging.debug(f"Encoded label '{target_col}' with classes: {le.classes_}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
logging.debug(f"After SMOTE, training set shape: {X_res.shape}")

# Train XGBoost classifier
clf = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
)
clf.fit(X_res, y_res)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.debug(f"Model training completed! Accuracy: {accuracy:.4f}")

# Save model and label encoder
joblib.dump(clf, "multi_disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")
logging.debug("Model and label encoder saved successfully!")
