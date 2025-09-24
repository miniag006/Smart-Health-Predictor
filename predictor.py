import pandas as pd
import joblib
from symptoms import SymptomMapper

# Load trained model
model = joblib.load("multi_disease_model.pkl")

# Initialize symptom mapper
symptom_mapper = SymptomMapper(csv_path="./datasets/Symptom-severity.csv")

# Example: symptoms reported by a patient
# Replace this list with actual input
patient_symptoms = [
    "headache", "fever", "nausea", "fatigue", "joint_pain"
]

# Convert symptoms to model input vector
all_symptoms = list(symptom_mapper.symptom_to_weight.keys())
input_vector = [
    symptom_mapper.get_weight(symptom) if symptom in patient_symptoms else 0
    for symptom in all_symptoms
]

# Convert to DataFrame with single row
X_new = pd.DataFrame([input_vector], columns=all_symptoms)

# Make prediction
predictions = model.predict(X_new)

# Map back encoded labels to readable names
# Each output label has its own encoder saved inside the MultiOutputClassifier
label_names = ["Class", "Outcome", "disease", "heart_disease", "fast_heart_rate"]

for i, col in enumerate(label_names):
    # Check if the classifier has classes_ attribute for each target
    if hasattr(model, "estimators_"):
        try:
            # MultiOutputClassifier stores each estimator for each label
            le_classes = model.estimators_[i].classes_
            pred_value = le_classes[predictions[0][i]]
            print(f"{col}: {pred_value}")
        except Exception:
            # fallback: print raw value
            print(f"{col}: {predictions[0][i]}")
    else:
        print(f"{col}: {predictions[0][i]}")
