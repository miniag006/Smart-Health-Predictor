import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.DEBUG)

# Load dataset
df = pd.read_csv('datasets/master_dataset.csv')
logging.debug(f"Master dataset shape: {df.shape}")

# Remove classes with <2 samples
y_counts = df['disease'].value_counts()
valid_classes = y_counts[y_counts >= 2].index
df = df[df['disease'].isin(valid_classes)]
logging.debug(f"Dataset shape after removing rare classes: {df.shape}")

# Separate features and target
X = df.drop(columns=['disease'])
y = df['disease']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Impute missing categorical values
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Scale numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
if numeric_cols:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logging.debug(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
logging.debug(f"Training shape after SMOTE: {X_train.shape}")

# Train RandomForest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")
