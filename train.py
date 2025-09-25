import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = 'datasets/master_dataset.csv'
logging.debug(f"Dataset found at: {dataset_path}")
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# Separate features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to string (if not already)
X[categorical_cols] = X[categorical_cols].astype(str)

# Remove completely empty categorical columns
non_empty_cols = [col for col in categorical_cols if X[col].notna().any()]

# Impute categorical columns if available
if non_empty_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[non_empty_cols] = cat_imputer.fit_transform(X[non_empty_cols])
else:
    logging.debug("No categorical columns to impute.")

# Impute numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
num_imputer = SimpleImputer(strategy='mean')
if numeric_cols:
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Encode categorical columns
for col in non_empty_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target labels
y = LabelEncoder().fit_transform(y.astype(str))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")
