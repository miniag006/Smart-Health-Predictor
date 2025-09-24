import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from symptom_mapper import SymptomMapper
import logging

logging.basicConfig(level=logging.DEBUG)

# Load model
clf = joblib.load("multi_disease_model.pkl")
logging.debug("Model loaded successfully!")

# Load symptoms
symptom_mapper = SymptomMapper()
all_symptoms = symptom_mapper.list_symptoms(n=len(symptom_mapper.symptom_to_weight))

# Format symptoms: replace underscores and capitalize each word
def format_symptom(s):
    return ' '.join(word.capitalize() for word in s.replace("_", " ").split())

formatted_symptoms = [format_symptom(s) for s in all_symptoms]

# Create main window
root = tk.Tk()
root.title("Smart Health Predictor")
root.geometry("900x700")

# Frames
top_frame = tk.Frame(root)
top_frame.pack(pady=10)
symptom_frame = tk.LabelFrame(root, text="Symptoms (Tick the checkboxes of the symptoms you have)")
symptom_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
numerical_frame = tk.LabelFrame(root, text="Numerical Input (Leave empty if do not have the data)")
numerical_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Search bar
search_var = tk.StringVar()
tk.Label(top_frame, text="Search Symptom:").pack(side=tk.LEFT, padx=5)
search_entry = tk.Entry(top_frame, textvariable=search_var, width=30)
search_entry.pack(side=tk.LEFT, padx=5)

# Symptoms checkboxes
symptom_vars = {}
symptom_checkbuttons = {}

def update_checkboxes(*args):
    query = search_var.get().lower()
    for s, cb in symptom_checkbuttons.items():
        if query in s.lower():
            cb.pack(anchor='w')
        else:
            cb.pack_forget()

search_var.trace_add('write', update_checkboxes)

for s in formatted_symptoms:
    var = tk.IntVar()
    cb = tk.Checkbutton(symptom_frame, text=s, variable=var, anchor='w', justify='left')
    cb.pack(anchor='w')
    symptom_vars[s] = var
    symptom_checkbuttons[s] = cb

# Numerical inputs
numerical_features = [
    "Age (years)", "Blood Pressure (mmHg)", "Albumin (g/dL)",
    "Weight (kg)", "Glucose (mg/dL)", "BMI", "Insulin (uIU/mL)"
]

numerical_vars = {}
for nf in numerical_features:
    frame = tk.Frame(numerical_frame)
    frame.pack(fill='x', pady=2)
    tk.Label(frame, text=nf+":", width=25, anchor='w').pack(side=tk.LEFT)
    var = tk.StringVar()
    tk.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT)
    numerical_vars[nf] = var

# Results table
result_frame = tk.Frame(root)
result_frame.pack(pady=10, fill='x')
tree = ttk.Treeview(result_frame, columns=("Disease", "Probability"), show='headings')
tree.heading("Disease", text="Disease")
tree.heading("Probability", text="Probability (%)")
tree.pack(fill='x')

# Predict button
def predict():
    # Prepare feature vector
    input_data = {}
    
    # Symptoms
    for s, var in symptom_vars.items():
        original_s = s.lower().replace(" ", "_")
        input_data[original_s] = var.get()
    
    # Numerical
    for nf, var in numerical_vars.items():
        col_name = nf.split()[0].lower()  # match your model columns if needed
        val = var.get()
        if val.strip() != "":
            try:
                input_data[col_name] = float(val)
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a numeric value for {nf}")
                return
    
    X = pd.DataFrame([input_data])
    
    try:
        # Predict probabilities
        y_probs = clf.predict_proba(X)
        # Clear previous results
        for row in tree.get_children():
            tree.delete(row)
        # For multi-output, clf.classes_ is a list
        if isinstance(clf.classes_[0], list) or isinstance(clf.classes_[0], np.ndarray):
            for i, col in enumerate(clf.classes_):
                for cls_index, cls in enumerate(col):
                    tree.insert('', 'end', values=(cls, round(y_probs[i][0][cls_index]*100, 2)))
        else:
            for cls_index, cls in enumerate(clf.classes_):
                tree.insert('', 'end', values=(cls, round(y_probs[0][cls_index]*100, 2)))
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

tk.Button(root, text="Predict", command=predict, bg="lightblue", font=('Arial', 12, 'bold')).pack(pady=5)

root.mainloop()
