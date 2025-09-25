import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.DEBUG)

# Load dataset
df = pd.read_csv('datasets/master_dataset.csv')
logging.debug(f"Master dataset shape: {df.shape}")

# Split features and target
X = df.drop(columns=['disease'])
y = df['disease']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=[np.number]).columns

# Impute missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')

if len(categorical_cols) > 0:
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
if len(numeric_cols) > 0:
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split without stratify first (to avoid single-sample class error)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Apply SMOTE on training set to balance all classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy:.4f}")
logging.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
