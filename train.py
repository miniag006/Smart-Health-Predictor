import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = 'datasets/master_dataset.csv'
logging.debug(f"Dataset found at: {dataset_path}")
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# Separate features and target
target_col = 'disease'
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target
y_encoded = LabelEncoder().fit_transform(y)
logging.debug(f"Label encoder classes saved: {list(LabelEncoder().fit(y).classes_)}")

# Handle missing values
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# Numeric columns: fill with mean
num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Categorical columns: convert to string, fill with most frequent, then encode
X[categorical_cols] = X[categorical_cols].astype(str)
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

logging.debug("Missing values handled for numeric and categorical columns")

# Feature selection: top 100 features
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y_encoded)
logging.debug("Selected top 100 features")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
try:
    X_resampled, y_resampled = smote.fit_resample(X_selected, y_encoded)
    logging.debug("Applied SMOTE to balance classes")
except ValueError as e:
    logging.warning(f"SMOTE skipped due to: {e}")
    X_resampled, y_resampled = X_selected, y_encoded

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
