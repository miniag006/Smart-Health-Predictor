import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = 'datasets/master_dataset.csv'
logging.debug(f"Dataset found at: {dataset_path}")
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# Separate features and target
label_col = 'disease'
X = df.drop(columns=[label_col])
y = df[label_col]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
logging.debug(f"Label encoder classes saved: {le.classes_}")

# Handle missing values
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include='object').columns

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Encode categorical features
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

logging.debug("Missing values handled for numeric and categorical columns")

# Feature selection
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y_encoded)
logging.debug("Selected top 100 features")

# Filter rare classes (appear less than 2 times)
counts = Counter(y_encoded)
rare_classes = [cls for cls, count in counts.items() if count < 2]
mask = ~np.isin(y_encoded, rare_classes)
X_filtered = X_selected[mask]
y_filtered = y_encoded[mask]

# SMOTE for balancing
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Accuracy
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
