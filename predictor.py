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
input_vector = []

for symptom in all_symptoms:
    # Use weight if symptom reported, else 0
    input_vector.append(symptom_mapper.get_weight(symptom) if symptom in patient_symptoms else 0)

# Convert to DataFrame with single row
X_new = pd.DataFrame([input_vector], columns=all_symptoms)

# Make prediction
predictions = model.predict(X_new)

# Output
disease_labels = model.classes_ if hasattr(model, "classes_") else None

# MultiOutputClassifier returns a 2D array
for idx, col in enumerate(["Class", "Outcome", "disease", "heart_disease", "fast_heart_rate"]):
    print(f"{col}: {predictions[0][idx]}")
