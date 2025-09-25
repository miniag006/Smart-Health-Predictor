import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained artifacts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_disease(user_inputs: dict):
    """
    Predict disease dynamically based on user input.

    user_inputs: dict
        Format example:
        {
            "fever": 1,
            "cough": 0,
            "age": 25,
            "bp": 120,
            ...
        }
    """
    # Convert user inputs to DataFrame (ensures any number of features)
    input_df = pd.DataFrame([user_inputs])

    # Fill missing features with 0 (or median if you prefer)
    for col in selector.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0  # default for unchecked checkbox / missing input

    # Keep only selected features
    input_selected = input_df[selector.feature_names_in_]

    # Scale the features
    input_scaled = scaler.transform(input_selected)

    # Predict using the model
    pred_class = model.predict(input_scaled)

    # Decode label
    pred_label = label_encoder.inverse_transform(pred_class)

    return pred_label[0]

# Example usage
if __name__ == "__main__":
    sample_input = {
        "fever": 1,
        "cough": 0,
        "fatigue": 1,
        "age": 30
    }
    prediction = predict_disease(sample_input)
    print("Predicted disease:", prediction)
