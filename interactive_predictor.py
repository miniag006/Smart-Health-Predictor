import sys
import os
import logging
import joblib
import pandas as pd
from symptoms import SymptomMapper
from tkinter import Tk, Label, Entry, Button, Checkbutton, IntVar, Frame, Scrollbar, Canvas
from tkinter import ttk

logging.basicConfig(level=logging.DEBUG)

# Ensure current folder is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
try:
    clf = joblib.load("multi_disease_model.pkl")
    logging.debug("Model loaded successfully!")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    exit()

# Initialize SymptomMapper
symptom_mapper = SymptomMapper()
symptoms = list(symptom_mapper.symptom_to_weight.keys())
logging.debug(f"Loaded {len(symptoms)} symptoms")

# Sample numerical features (replace with your dataset's full columns)
numerical_features = ["Age (years)", "Blood Pressure (mmHg)", "Albumin (g/dL)",
                      "Weight (kg)", "Glucose (mg/dL)", "BMI", "Cholesterol (mg/dL)"]

# Main window
root = Tk()
root.title("Smart Health Predictor")
root.geometry("900x700")

# ===== Scrollable Frame Setup =====
canvas = Canvas(root)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ===== Symptoms Section =====
Label(scrollable_frame, text="Symptoms (Tick the checkboxes of the symptoms you have)", font=("Arial", 12, "bold")).pack(pady=5)

search_var = Entry(scrollable_frame, width=50)
search_var.pack(pady=5)
search_var.insert(0, "Search symptom...")

# Container for symptom checkboxes
symptom_frame = Frame(scrollable_frame)
symptom_frame.pack()

symptom_vars = {}
def update_symptoms(*args):
    for widget in symptom_frame.winfo_children():
        widget.destroy()
    query = search_var.get().lower()
    for s in symptoms:
        if query in s.lower():
            var = symptom_vars.get(s, IntVar())
            symptom_vars[s] = var
            cb = Checkbutton(symptom_frame, text=s.replace("_", " ").title(), variable=var)
            cb.pack(anchor="w")

search_var.bind("<KeyRelease>", update_symptoms)
update_symptoms()

# ===== Numerical Inputs Section =====
Label(scrollable_frame, text="Numerical Inputs (Leave empty if not available)", font=("Arial", 12, "bold")).pack(pady=5)

num_entries = {}
for feature in numerical_features:
    frame = Frame(scrollable_frame)
    frame.pack(pady=2, fill="x")
    Label(frame, text=feature, width=25, anchor="w").pack(side="left")
    entry = Entry(frame, width=20)
    entry.pack(side="left")
    num_entries[feature] = entry

# ===== Predict Button and Results =====
def predict():
    user_data = {}
    # Symptoms
    for s, var in symptom_vars.items():
        if var.get() == 1:
            user_data[s] = 1
    # Numerical inputs
    for f, entry in num_entries.items():
        val = entry.get().strip()
        if val != "":
            try:
                if "." in val:
                    user_data[f] = float(val)
                else:
                    user_data[f] = int(val)
            except:
                logging.warning(f"Invalid input for {f}, ignoring.")

    if not user_data:
        logging.warning("No input provided")
        return

    # Convert to DataFrame
    df = pd.DataFrame([user_data])
    # Ensure missing columns in df are ignored by the model
    missing_cols = set(clf.feature_names_in_) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Predict probabilities
    try:
        probs = clf.predict_proba(df)
        result_frame = Frame(scrollable_frame)
        result_frame.pack(pady=10)
        Label(result_frame, text="Predicted Diseases with Probability", font=("Arial", 12, "bold")).pack()
        table = ttk.Treeview(result_frame, columns=("Disease", "Probability"), show="headings")
        table.heading("Disease", text="Disease")
        table.heading("Probability", text="Probability (%)")
        table.pack()

        # For multi-output, probs is list of arrays
        for i, classes in enumerate(clf.classes_):
            for cls, prob in zip(classes, probs[i][0]):
                table.insert("", "end", values=(cls, f"{prob*100:.2f}%"))
    except Exception as e:
        logging.error(f"Prediction failed: {e}")

Button(scrollable_frame, text="Predict", command=predict, bg="green", fg="white", width=20).pack(pady=10)

root.mainloop()
