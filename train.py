import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.DEBUG)

# -------------------- Paths --------------------
dataset_path = "datasets/master_dataset.csv"  # CSV file path
model_path = "multidisease_model.pkl"
label_encoder_path = "label_encoder.pkl"

# -------------------- Load dataset --------------------
logging.debug(f"Dataset found at: {dataset_path}")
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# -------------------- Define label --------------------
label_column = 'disease'  # Use 'disease' as target
y = df[label_column]
X = df.drop(label_column, axis=1)

# -------------------- Encode target --------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, label_encoder_path)
logging.debug(f"Label encoder classes saved: {list(label_encoder.classes_)}")

# -------------------- Handle missing values --------------------
# Separate numeric and categorical
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# Impute numeric with median
num_imputer = SimpleImputer(strategy='median')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Impute categorical with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

logging.debug("Missing values handled for numeric and categorical columns")

# -------------------- Encode categorical variables --------------------
X = pd.get_dummies(X, columns=categorical_cols)
logging.debug(f"Shape after one-hot encoding: {X.shape}")

# -------------------- Feature selection --------------------
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y_encoded)
logging.debug(f"Selected top 100 features")

# -------------------- SMOTE oversampling --------------------
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y_encoded)
logging.debug(f"Resampled dataset shape: {X_resampled.shape}, num_classes={len(np.unique(y_resampled))}")

# -------------------- Train XGBoost --------------------
model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_resampled)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_resampled, y_resampled)
logging.debug("Model training completed")

# -------------------- Save model --------------------
joblib.dump(model, model_path)
logging.debug(f"Model saved to {model_path}")
