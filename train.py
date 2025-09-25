import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load dataset
df = pd.read_csv("datasets/master_dataset.csv")
logging.debug(f"Master dataset shape: {df.shape}")

# Target column
target_col = 'disease'

# Remove classes with less than 2 samples (SMOTE requirement)
class_counts = df[target_col].value_counts()
rare_classes = class_counts[class_counts < 2].index
if len(rare_classes) > 0:
    logging.debug(f"Removing rare classes with <2 samples: {list(rare_classes)}")
    df = df[~df[target_col].isin(rare_classes)]

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Impute missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
if categorical_cols:
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
num_imputer = SimpleImputer(strategy='mean')
if num_cols:
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Scale numeric features
scaler = StandardScaler()
if num_cols:
    X[num_cols] = scaler.fit_transform(X[num_cols])

# Encode categorical features if needed (e.g., one-hot)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)  # k_neighbors=1 to handle small classes
X_train, y_train = smote.fit_resample(X_train, y_train)
logging.debug(f"Resampled X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.debug(f"Test Accuracy: {accuracy:.4f}")
logging.debug("Classification Report:\n" + classification_report(y_test, y_pred))

# Save model and label encoder if needed
import joblib
joblib.dump(model, "models/random_forest_model.pkl")
logging.debug("Model saved to models/random_forest_model.pkl")
