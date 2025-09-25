import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.DEBUG)

# Load dataset
dataset_path = r"E:\Projects\Smart Health Predictor\Smart-Health-Predictor\datasets\master_dataset.csv"


logging.debug(f"Dataset found at: {dataset_path}")
df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# Split features and labels
X = df.drop('label', axis=1)  # replace 'label' with your actual label column name
y = df['label']

# Encode labels if they are categorical
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, 'models/label_encoder.pkl')
logging.debug(f"Label encoder classes saved: {le.classes_}")

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
logging.debug("Missing values handled with median imputation")

# Feature selection: select top 100 features
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_imputed, y)
logging.debug("Selected top 100 features")

# SMOTE with k_neighbors=1
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)
logging.debug(f"Resampled dataset shape: {X_resampled.shape}")
logging.debug(f"Post-SMOTE label classes: {list(set(y_resampled))}, num_classes={len(set(y_resampled))}")

# Train XGBoost classifier
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_resampled, y_resampled)
logging.debug("Model training complete")

# Save the trained model
joblib.dump(model, 'models/xgb_model.pkl')
logging.debug("Trained model saved as 'models/xgb_model.pkl'")
