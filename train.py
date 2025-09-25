import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = "datasets/master_dataset.csv"
df = pd.read_csv(dataset_path)
logging.debug(f"Dataset found at: {dataset_path}")
logging.debug(f"Master dataset shape: {df.shape}")

# Features and target
X = df.drop(columns=['disease'])
y = df['disease']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Impute missing values for categorical columns
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    non_empty_cols = [col for col in categorical_cols if not X[col].empty]
    if non_empty_cols:
        X[non_empty_cols] = cat_imputer.fit_transform(X[non_empty_cols])
        logging.debug(f"Categorical columns after imputation: {non_empty_cols}")

# Impute missing values for numerical columns
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
if numerical_cols:
    num_imputer = SimpleImputer(strategy='mean')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance with SMOTE if training set has more than 1 class
unique_classes = np.unique(y_train)
if len(unique_classes) > 1:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logging.debug(f"After SMOTE, X_train shape: {X_train.shape}, y_train classes: {np.unique(y_train)}")
else:
    logging.debug("Skipping SMOTE because training set has only 1 class")

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
