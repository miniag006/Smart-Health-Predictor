import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

# Load master dataset
df = pd.read_csv(r"dataset/master_dataset.csv")  # <-- your master dataset path
logging.debug(f"Master dataset shape: {df.shape}")

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(df['disease'])
logging.debug(f"Encoded label 'disease' with classes: {le.classes_}")

# Features
X = df.drop(['disease'], axis=1)

# Handle missing values
X = X.fillna(X.median())
logging.debug("Missing values handled with median imputation")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_selected = selector.fit_transform(X_scaled, y_encoded)
logging.debug("Selected top 50 features")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)  # <- fix for small classes
X_res, y_res = smote.fit_resample(X_selected, y_encoded)
logging.debug(f"Resampled dataset shape: {X_res.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.debug(f"Model training completed! Accuracy: {accuracy:.4f}")

# Save model, encoder, scaler, selector
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')
logging.debug("Model, label encoder, scaler, and selector saved successfully!")
