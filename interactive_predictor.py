import sys
import os
import logging
import joblib
import pandas as pd
from symptoms import SymptomMapper
from tkinter import Tk, Label, Entry, Button, Checkbutton, IntVar, Frame, Canvas
from tkinter import ttk

logging.basicConfig(level=logging.DEBUG)

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

# Define numerical features with units
numerical_features = [
    "Age (years)", "Blood Pressure (mmHg)", "Albumin (g/dL)", "Weight (kg)",
    "Glucose (mg/dL)", "BMI", "Cholesterol (mg/dL)", "Creatinine (mg/dL)",
    "Heart Rate (bpm)", "Hemoglobin (g/dL)", "Sodium (mmol/L)", "Potassium (mmol/L)",
    "Calcium (mg/dL)", "Magnesium (mg/dL)", "Phosphorus (mg/dL)", "Urea (mg/dL)",
    "Albumin/Globulin Ratio", "Total Protein (g/dL)", "Alkaline Phosphatase (IU/L)",
    "Bilirubin (mg/dL)", "AST (SGOT) (IU/L)", "ALT (SGPT) (IU/L)", "WBC Count (cells/μL)",
    "RBC Count (million cells/μL)", "Platelet Count (cells/μL)", "Hematocrit (%)",
    "MCV (fL)", "MCH (pg)", "MCHC (g/dL)", "RDW (%)", "Neutrophil Count (%)",
    "Lymphocyte Count (%)", "Monocyte Count (%)", "Eosinophil Count (%)", "Basophil Count (%)",
    "Serum Iron (μg/dL)", "Ferritin (ng/mL)", "Transferrin Saturation (%)", "TIBC (μg/dL)",
    "LDH (IU/L)", "CRP (mg/L)", "ESR (mm/hr)"
]

# --- UI ---
root = Tk()
root.title("Smart Health Predictor")
root.geometry("1000x700")

# Scrollable canvas (no visible scrollbar)
canvas = Canvas(root)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.pack(side="left", fill="both", expand=True)

# Enable scrolling with mouse wheel
def _on_mouse_wheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

# --- Top Bar with Predict Button ---
top_bar = Frame(scrollable_frame)
top_bar.pack(fill="x", pady=10)

def predict():
    selected_symptoms = [symptom for symptom, var in symptom_vars.items() if var.get() == 1]
    input_data = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in symptoms}

    for feature, entry in numerical_entries.items():
        value = entry.get().strip()
        if value:
            try:
                input_data[feature] = float(value)
            except ValueError:
                input_data[feature] = 0
        else:
            input_data[feature] = 0

    df_input = pd.DataFrame([input_data])
    probabilities = clf.predict_proba(df_input)[0]
    disease_probs = {clf.classes_[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}

    for row in result_table.get_children():
        result_table.delete(row)

    for disease, prob in sorted(disease_probs.items(), key=lambda x: x[1], reverse=True):
        result_table.insert("", "end", values=(disease, f"{prob}%"))

predict_btn = Button(top_bar, text="Predict", command=predict,
                     bg="lightgreen", font=("Arial", 12, "bold"))
predict_btn.pack(side="left", padx=20)

# --- Results Table (Right below Predict button) ---
result_table = ttk.Treeview(scrollable_frame, columns=("Disease", "Probability"), show="headings", height=10)
result_table.heading("Disease", text="Disease")
result_table.heading("Probability", text="Probability")
result_table.pack(fill="x", padx=20, pady=10)

# --- Two Columns Layout ---
columns_frame = Frame(scrollable_frame)
columns_frame.pack(pady=10, fill="x")

# Symptoms Column
symptom_frame = Frame(columns_frame)
symptom_frame.pack(side="left", padx=20, anchor="n")

Label(symptom_frame, text="Symptoms (Tick the checkboxes of the symptoms you have)", 
      font=("Arial", 10, "bold")).pack(anchor="w")

search_symptom = Entry(symptom_frame)
search_symptom.pack(pady=5, fill="x")

symptom_vars = {}
checkboxes = []

def update_symptom_list(*args):
    search_text = search_symptom.get().lower()
    for cb, symptom in checkboxes:
        if search_text in symptom.lower():
            cb.pack(anchor="w")
        else:
            cb.pack_forget()

for symptom in symptoms:
    var = IntVar()
    cb = Checkbutton(symptom_frame, text=symptom.replace("_", " ").title(), variable=var)
    symptom_vars[symptom] = var
    checkboxes.append((cb, symptom))
    cb.pack(anchor="w")

search_symptom.bind("<KeyRelease>", update_symptom_list)

# Numerical Inputs Column
num_frame = Frame(columns_frame)
num_frame.pack(side="left", padx=20, anchor="n")

Label(num_frame, text="Numerical Input (Leave empty if not available)", 
      font=("Arial", 10, "bold")).pack(anchor="w")

search_num = Entry(num_frame)
search_num.pack(pady=5, fill="x")

numerical_entries = {}
num_widgets = []

def update_num_list(*args):
    search_text = search_num.get().lower()
    for frame, feature, entry in num_widgets:
        if search_text in feature.lower():
            frame.pack(fill="x", pady=2)
        else:
            frame.pack_forget()

for feature in numerical_features:
    frame = Frame(num_frame)
    frame.pack(fill="x", pady=2)
    Label(frame, text=feature, width=30, anchor="w").pack(side="left")
    entry = Entry(frame)
    entry.pack(side="left", fill="x", expand=True)
    numerical_entries[feature] = entry
    num_widgets.append((frame, feature, entry))

search_num.bind("<KeyRelease>", update_num_list)

root.mainloop()
