import pandas as pd
import joblib
from symptoms import SymptomMapper  # your updated symptoms.py
from sklearn.preprocessing import LabelEncoder

# Initialize SymptomMapper
symptom_mapper = SymptomMapper(csv_path="./datasets/Symptom-severity.csv")

# Load the trained model
clf = joblib.load("multi_disease_model.pkl")

# Load training dataset to get feature columns and label encoders
train_data = pd.read_csv("./datasets/training_data_git_2.csv")
feature_columns = [col for col in train_data.columns if col != 'fast_heart_rate']  # exclude label

# Build LabelEncoders for each label column
label_columns = ['fast_heart_rate']  # add all your label columns here if multioutput
label_encoders = {}
for col in label_columns:
    le = LabelEncoder()
    le.fit(train_data[col].fillna("Unknown").astype(str))
    label_encoders[col] = le

# Function to convert symptom list to model input
def symptoms_to_input(symptoms, all_features):
    """
    Convert a list of symptom names into a DataFrame row for prediction.
    'all_features' must match training columns.
    """
    data = {feature: 0 for feature in all_features}
    for symptom in symptoms:
        if symptom in data:
            data[symptom] = symptom_mapper.get_weight(symptom)
    return pd.DataFrame([data])

# Function to decode predictions into readable labels
def decode_prediction(pred_array):
    decoded = {}
    for idx, col in enumerate(label_columns):
        le = label_encoders[col]
        decoded[col] = le.inverse_transform([pred_array[0][idx]])[0]  # decode single row
    return decoded

# Example usage
if __name__ == "__main__":
    user_symptoms = ["fever", "headache", "nausea"]
    X_input = symptoms_to_input(user_symptoms, feature_columns)
    
    # Predict
    prediction = clf.predict(X_input)
    
    # Decode to readable output
    readable_output = decode_prediction(prediction)
    
    print("Predicted conditions/diseases:")
    for label, value in readable_output.items():
        print(f"{label}: {value}")
