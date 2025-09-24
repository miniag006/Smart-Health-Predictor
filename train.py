import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.DEBUG)

# Load master dataset
df = pd.read_csv("./datasets/master_dataset.csv")
logging.debug(f"Master dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=["disease"])  # assuming 'disease' is your target column
y = df["disease"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
logging.debug(f"Model training completed! Accuracy: {accuracy:.4f}")

# Save model and label encoder
joblib.dump(clf, "multi_disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")
logging.debug("Model and label encoder saved successfully!")
