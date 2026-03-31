import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import json, pickle, os
import joblib


MODELS_PATH = "models/models_bayesian.joblib"
RF_PATH = "models/random_forest_meta.joblib"
CLASSES_PATH = "models/classes_order.json"
METRICS_PATH = "models/metrics.json"


if not os.path.exists(MODELS_PATH) or not os.path.exists(RF_PATH):
    messagebox.showerror("Error", "Trained models not found. Please run train.py first.")
    exit()


models = joblib.load(MODELS_PATH)
rf = joblib.load(RF_PATH)


with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)
    accuracy = metrics.get("accuracy", 0) * 100


TRAIN_PATH = "data/prepared_train.csv"
if not os.path.exists(TRAIN_PATH):
    messagebox.showerror("Error", "Training data not found.")
    exit()

train_df = pd.read_csv(TRAIN_PATH)
features = [col for col in train_df.columns if col != "prognosis"]


root = tk.Tk()
root.title("🩺 Smart Health Predictor — Hybrid Bayesian + RandomForest")
root.geometry("900x700")
root.config(bg="#f0f7ff")

tk.Label(
    root,
    text="Smart Health Predictor",
    font=("Arial", 26, "bold"),
    bg="#f0f7ff",
    fg="#0077b6"
).pack(pady=20)

tk.Label(
    root,
    text=f"Model Accuracy: {accuracy:.2f}%",
    font=("Arial", 14),
    bg="#f0f7ff"
).pack()


canvas = tk.Canvas(root, bg="#f0f7ff", highlightthickness=0)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
frame = tk.Frame(canvas, bg="#f0f7ff")

symptom_vars = {}
for i, symptom in enumerate(features):
    var = tk.IntVar()
    chk = tk.Checkbutton(frame, text=symptom, variable=var, bg="#f0f7ff", anchor="w")
    chk.grid(row=i, column=0, sticky="w", padx=20)
    symptom_vars[symptom] = var

canvas.create_window((0, 0), window=frame, anchor="nw")
canvas.update_idletasks()
canvas.configure(scrollregion=canvas.bbox("all"), yscrollcommand=scroll_y.set)
canvas.pack(fill="both", expand=True, side="left")
scroll_y.pack(fill="y", side="right")



def predict_disease():
    print("Predict button clicked")

    input_values = [symptom_vars[s].get() for s in features]
    if sum(input_values) == 0:
        messagebox.showwarning("No Symptoms Selected", "Please select at least one symptom.")
        return

    import joblib, os, json  

    input_data = np.array(input_values).reshape(1, -1)

    try:


        
        models_path = "models/models_bayesian.joblib"
        rf_path = "models/random_forest_meta.joblib"
        scaler_path = "models/score_scaler.joblib"
        classes_path = "models/classes_order.json"

       
        for path in [models_path, rf_path, scaler_path, classes_path]:
            if not os.path.exists(path):
                messagebox.showerror("Error", f"Missing file: {path}\nPlease retrain the model.")
                return

        
        models = joblib.load(models_path)
        rf = joblib.load(rf_path)
        score_scaler = joblib.load(scaler_path)
        with open(classes_path, "r") as f:
            classes = json.load(f)

        
        scores = np.column_stack([models[c].predict(input_data) for c in classes])

       
        scores = score_scaler.transform(scores)

        
        probs = rf.predict_proba(scores)[0]
        disease_probs = list(zip(rf.classes_, probs))
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        
        top5 = disease_probs[:5]
        total = sum([p for _, p in top5])
        if total > 0:
            top5 = [(d, p / total) for d, p in top5]  

        result_text = "\n".join([f"{d}: {p*100:.2f}%" for d, p in top5])

        print("Prediction debug:", result_text)
        messagebox.showinfo("Prediction Results", f"Top Predicted Diseases:\n\n{result_text}")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        messagebox.showerror("Error", f"Prediction failed:\n{e}")



tk.Button(
    root,
    text="Predict Disease",
    command=predict_disease,
    font=("Arial", 14, "bold"),
    bg="#00b4d8",
    fg="white",
    padx=20,
    pady=10
).pack(pady=15)

tk.Button(
    root,
    text="Exit",
    command=root.destroy,
    font=("Arial", 12, "bold"),
    bg="#ff4d6d",
    fg="white",
    padx=20,
    pady=8
).pack(pady=5)

root.mainloop()
