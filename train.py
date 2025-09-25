import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = "datasets/master_dataset.csv"
logging.debug(f"Dataset found at: {dataset_path}")
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# Identify features and target
label_col = 'prognosis'  # replace with actual label column name
X = df.drop(label_col, axis=1)
y = df[label_col]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'label_encoder.pkl')
logging.debug(f"Label encoder classes saved: {label_encoder.classes_}")

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Handle missing values
if numeric_cols:
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
logging.debug("Missing values handled for numeric and categorical columns")

# One-hot encode categorical features (only if any exist)
if categorical_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_categorical_encoded = pd.DataFrame(ohe.fit_transform(X[categorical_cols]),
                                         columns=ohe.get_feature_names_out(categorical_cols),
                                         index=X.index)
    X = pd.concat([X[numeric_cols], X_categorical_encoded], axis=1)
else:
    logging.debug("No categorical columns found, skipping One-Hot Encoding")

# Feature selection: select top 100 features
k_features = min(100, X.shape[1])  # avoid error if less than 100 features
selector = SelectKBest(score_func=f_classif, k=k_features)
X_selected = selector.fit_transform(X, y_encoded)
logging.debug(f"Selected top {k_features} features")

# Handle class imbalance using SMOTE with k_neighbors=1
smote = SMOTE(k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_selected, y_encoded)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.debug(f"Model Accuracy: {accuracy}")

# Save model
joblib.dump(model, 'multidisease_model.pkl')
logging.debug("Trained model saved as multidisease_model.pkl")
