import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = 'datasets/master_dataset.csv'
logging.debug(f'Dataset found at: {dataset_path}')
df = pd.read_csv(dataset_path)
logging.debug(f'Master dataset shape: {df.shape}')

# Target variable
target_col = 'disease'
y = df[target_col]
X = df.drop(columns=[target_col])

# Handle missing values
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# Numeric imputer
num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Categorical imputer
X[categorical_cols] = X[categorical_cols].astype(str)
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Encode categorical features
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

logging.debug('Missing values handled for numeric and categorical columns')

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
logging.debug(f'Label encoder classes saved: {label_encoder.classes_}')

# Feature selection (top 100)
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y_encoded)
logging.debug('Selected top 100 features')

# SMOTE: handle rare classes safely
y_series = pd.Series(y_encoded)
class_counts = y_series.value_counts()
# Keep only classes with at least 2 samples
valid_classes = class_counts[class_counts > 1].index
mask = y_series.isin(valid_classes)

X_valid = X_selected[mask]
y_valid = y_encoded[mask]

if len(np.unique(y_valid)) > 1:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_valid, y_valid)
    logging.debug('Applied SMOTE oversampling on valid classes')
else:
    logging.debug('Not enough classes for SMOTE, skipping')
    X_resampled, y_resampled = X_valid, y_valid

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions & accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {acc:.4f}')
