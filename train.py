import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = "datasets/master_dataset.csv"
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# Separate features and target
target_col = 'disease'
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Impute missing categorical values
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Impute missing numerical values
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
if numerical_cols:
    num_imputer = SimpleImputer(strategy='mean')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Remove rare classes globally before splitting
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
if len(rare_classes) > 0:
    logging.debug(f"Removing rare classes with <2 samples: {list(rare_classes)}")
    mask = ~y.isin(rare_classes)
    X_scaled = X_scaled[mask]
    y = y[mask]

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

logging.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Remove rare classes in training set for SMOTE
train_class_counts = y_train.value_counts()
rare_train_classes = train_class_counts[train_class_counts < 2].index
if len(rare_train_classes) > 0:
    logging.debug(f"Removing rare classes in training set: {list(rare_train_classes)}")
    mask_train = ~y_train.isin(rare_train_classes)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

# Apply SMOTE with k_neighbors=1 to avoid n_neighbors error
smote = SMOTE(random_state=42, k_neighbors=1)
X_train, y_train = smote.fit_resample(X_train, y_train)
logging.debug(f"After SMOTE: X_train {X_train.shape}, y_train {Counter(y_train)}")

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
logging.debug(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logging.debug(f"Classification Report:\n{classification_report(y_test, y_pred)}")
