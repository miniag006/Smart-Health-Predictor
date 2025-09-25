import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.DEBUG)

# --- Load dataset ---
dataset_path = r"E:\Projects\Smart Health Predictor\Smart-Health-Predictor\datasets\master_dataset.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
logging.debug(f"Dataset found at: {dataset_path}")

df = pd.read_csv(dataset_path)
logging.debug(f"Master dataset shape: {df.shape}")

# --- Encode target ---
le = LabelEncoder()
y_encoded = le.fit_transform(df['disease'])
logging.debug(f"Encoded label 'disease' with classes: {le.classes_}")

# --- Features ---
X = df.drop(['disease'], axis=1)
X = X.fillna(X.median())  # handle missing values
logging.debug("Missing values handled with median imputation")

# --- Feature scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Feature selection ---
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_scaled, y_encoded)
logging.debug("Selected top 100 features")

# --- Handle class imbalance with SMOTE safely ---
class_counts = pd.Series(y_encoded).value_counts()
eligible_classes = class_counts[class_counts > 1].index
mask = np.isin(y_encoded, eligible_classes)
X_filtered = X_selected[mask]
y_filtered = y_encoded[mask]

smote = SMOTE(random_state=42, k_neighbors=1)
X_res, y_res = smote.fit_resample(X_filtered, y_filtered)
logging.debug(f"Resampled dataset shape: {X_res.shape}")

# --- Stratified K-Fold ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_res, y_res), 1):
    X_train, X_test = X_res[train_idx], X_res[test_idx]
    y_train, y_test = y_res[train_idx], y_res[test_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_res)),
        'max_depth': 10,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    y_pred = bst.predict(dtest)
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)
    logging.debug(f"Fold {fold} Accuracy: {acc:.4f}")

logging.debug(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f}")

# --- Save artifacts ---
joblib.dump(bst, 'xgb_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')
logging.debug("Model, label encoder, scaler, and selector saved successfully!")
