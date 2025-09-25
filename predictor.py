import pandas as pd
import numpy as np
import joblib

# --- Load saved artifacts ---
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
le = joblib.load("label_encoder.pkl")

# Load the original dataset once (to know all features and median values)
dataset_path = r"E:\Projects\Smart Health Predictor\Smart-Health-Predictor\datasets\master_dataset.csv"
df = pd.read_csv(dataset_path)
all_features = df.drop(["disease"], axis=1).columns
feature_medians = df.drop(["disease"], axis=1).median()


# --- Function to prepare user input ---
def prepare_input(user_input: dict):
    """
    user_input: dict of {feature_name: value}
    Example: {"fever": 1, "cough": 1, "age": 35}
    """

    # Start with median values for all features
    input_data = feature_medians.copy()

    # Update only the features user provided
    for feature, value in user_input.items():
        if feature in input_data.index:
            input_data[feature] = value

    # Convert to DataFrame with one row
    input_df = pd.DataFrame([input_data])

    # Apply same scaling + feature selection as training
    X_scaled = scaler.transform(input_df)
    X_selected = selector.transform(X_scaled)

    return X_selected


# --- Function to predict disease ---
def predict_disease(user_input: dict):
    X_selected = prepare_input(user_input)
    prediction = model.predict(X_selected)
    disease = le.inverse_transform(prediction)[0]
    return disease


# --- Example Usage ---
if __name__ == "__main__":
    # Example: user selects 5 symptoms + 2 numerical inputs
    user_input = {
        "fever": 1,
        "cough": 1,
        "headache": 1,
        "fatigue": 1,
        "nausea": 1,
        "age": 30,
        "blood_pressure": 120
    }

    result = predict_disease(user_input)
    print("Predicted Disease:", result)
