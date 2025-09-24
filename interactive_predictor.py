import pandas as pd
import joblib
import logging
import tkinter as tk
from tkinter import ttk, messagebox

logging.basicConfig(level=logging.DEBUG)

# Load trained model
clf = joblib.load("multi_disease_model.pkl")
logging.debug("Model loaded successfully!")

# SymptomMapper class
class SymptomMapper:
    def __init__(self, csv_path="./datasets/Symptom-severity.csv"):
        logging.debug("Initializing SymptomMapper...")
        self.symptom_to_weight = {}
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.symptom_to_weight[row['Symptom']] = row['weight']
            logging.debug(f"Loaded {len(self.symptom_to_weight)} symptoms")
        except Exception as e:
            logging.error(f"Failed to load symptoms: {e}")

    def list_symptoms(self):
        return list(self.symptom_to_weight.keys())

symptom_mapper = SymptomMapper()
all_symptoms = symptom_mapper.list_symptoms()

# Numerical features with units
numerical_features = {
    'Age': 'years', 
    'Bp': 'mmHg', 
    'Sg': '', 
    'Al': '', 
    'Su': '', 
    'Rbc': '', 
    'Bu': 'mg/dL', 
    'Sc': 'mg/dL', 
    'Sod': 'mEq/L', 
    'Pot': 'mEq/L',
    'Hemo': 'g/dL', 
    'BMI': 'kg/m^2', 
    'Glucose': 'mg/dL', 
    'BloodPressure': 'mmHg', 
    'SkinThickness': 'mm', 
    'Insulin': 'μU/mL', 
    'DiabetesPedigreeFunction': '',
    # Add other numeric features if needed
}

# GUI window
root = tk.Tk()
root.title("Smart Health Predictor")
root.geometry("700x800")

tk.Label(root, text="Select your symptoms:").pack()

# Scrollable symptoms frame
canvas = tk.Canvas(root, height=250)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas)

scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Symptom checkboxes
symptom_vars = {}
for symptom in all_symptoms:
    var = tk.IntVar()
    cb = tk.Checkbutton(scroll_frame, text=symptom, variable=var)
    cb.pack(anchor='w')
    symptom_vars[symptom] = var

tk.Label(root, text="Enter numerical features (leave blank if not applicable):").pack(pady=5)
num_frame = tk.Frame(root)
num_frame.pack(fill=tk.BOTH, expand=True)

num_entries = {}
for feature, unit in numerical_features.items():
    frame = tk.Frame(num_frame)
    frame.pack(fill=tk.X, pady=2)
    tk.Label(frame, text=f"{feature} ({unit})", width=30, anchor='w').pack(side=tk.LEFT)
    entry = tk.Entry(frame)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    num_entries[feature] = entry

# Frame to display prediction results
results_frame = tk.Frame(root)
results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

tree = ttk.Treeview(results_frame, columns=("Disease", "Probability"), show='headings')
tree.heading("Disease", text="Disease")
tree.heading("Probability", text="Probability (%)")
tree.column("Disease", width=300)
tree.column("Probability", width=150)
tree.pack(fill=tk.BOTH, expand=True)

# Prediction function
def predict():
    tree.delete(*tree.get_children())
    selected_symptoms = [sym for sym, var in symptom_vars.items() if var.get() == 1]
    num_data = {}
    for feature, entry in num_entries.items():
        val = entry.get().strip()
        if val:
            try:
                num_data[feature] = float(val)
            except ValueError:
                messagebox.showerror("Invalid input", f"Value for {feature} must be numeric.")
                return

    X = pd.DataFrame([num_data], columns=num_data.keys())

    # Set selected symptoms to 1
    for symptom in selected_symptoms:
        X[symptom] = 1

    # Fill missing columns with 0
    for col in clf.feature_names_in_:
        if col not in X.columns:
            X[col] = 0

    X = X[clf.feature_names_in_]

    try:
        probs = clf.predict_proba(X)
        diseases = clf.classes_
        # For multi-output, combine probabilities
        if isinstance(probs, list):
            results = []
            for i, p in enumerate(probs):
                for j, disease in enumerate(diseases[i]):
                    results.append((disease, round(p[j]*100, 2)))
        else:
            results = [(diseases[i], round(probs[0][i]*100, 2)) for i in range(len(diseases))]

        results.sort(key=lambda x: x[1], reverse=True)

        # Insert into treeview
        for disease, prob in results:
            tree.insert("", tk.END, values=(disease, prob))
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

tk.Button(root, text="Predict", command=predict, bg="green", fg="white").pack(pady=10)
root.mainloop()
