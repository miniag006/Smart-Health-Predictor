import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# 1️⃣ Dataset info: file names and label columns
datasets_info = {
    "chronic_kidney_disease.csv": ["Class"],
    "diabetes.csv": [],  # No label column
    "diabetes_2.csv": ["Outcome"],
    "disease.csv": ["disease"],
    "heart_disease.csv": [],  # No label column
    "heart_disease_2.csv": ["heart_disease"],
    "kidney_2.csv": ["fast_heart_rate"]
}

# 2️⃣ Initialize sets
all_symptoms = set()
all_numerical = set()
all_labels = set()
dataframes = []

# 3️⃣ Read all datasets and collect features
for file, labels in datasets_info.items():
    path = os.path.join("datasets", file)
    if not os.path.exists(path):
        logging.warning(f"{path} not found, skipping.")
        continue

    df = pd.read_csv(path)
    logging.debug(f"Loaded {file} with shape {df.shape}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_numerical.update(numeric_cols)

    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
    symptom_cols = [c for c in categorical_cols if c not in labels]
    all_symptoms.update(symptom_cols)

    all_labels.update(labels)
    dataframes.append(df)

# 4️⃣ Standardize column names
def standardize_name(name):
    return name.replace("_", " ").title()

all_symptoms = [standardize_name(s) for s in all_symptoms]
all_numerical = [standardize_name(n) for n in all_numerical]
all_labels = [standardize_name(l) for l in all_labels]

logging.debug(f"Total symptoms: {len(all_symptoms)}")
logging.debug(f"Total numerical features: {len(all_numerical)}")
logging.debug(f"Total label columns: {len(all_labels)}")

# 5️⃣ Convert datasets to master format
master_rows = []

for df in dataframes:
    row_dict = {}

    # Symptoms -> binary 1/0
    for s in all_symptoms:
        orig_col = s.lower().replace(" ", "_")
        if orig_col in df.columns:
            row_dict[s] = df[orig_col].apply(lambda x: 1 if str(x).strip().lower() not in ["0", "no", "nan", ""] else 0)
        else:
            row_dict[s] = 0

    # Numerical
    for n in all_numerical:
        orig_col = n.lower().replace(" ", "_")
        if orig_col in df.columns:
            row_dict[n] = df[orig_col]
        else:
            row_dict[n] = np.nan

    # Labels
    for l in all_labels:
        orig_col = l.lower().replace(" ", "_")
        if orig_col in df.columns:
            row_dict[l] = df[orig_col]
        else:
            row_dict[l] = 0

    master_rows.append(pd.DataFrame(row_dict))

# 6️⃣ Concatenate all rows
if master_rows:
    master_df = pd.concat(master_rows, ignore_index=True)
    # 7️⃣ Save master dataset
    master_df.to_csv("master_dataset.csv", index=False)
    logging.info("Master dataset created successfully! Saved as master_dataset.csv")
else:
    logging.error("No datasets loaded. Master dataset not created.")
