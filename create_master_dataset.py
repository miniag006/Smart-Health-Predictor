import os
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

# Folder where all datasets are stored
dataset_folder = "./datasets"

# List of CSV files to combine
dataset_files = [
    "chronic_kidney_disease.csv",
    "diabetes.csv",
    "diabetes_2.csv",
    "disease.csv",
    "heart_disease.csv",
    "heart_disease_2.csv",
    "kidney_2.csv"
]

# Initialize containers
all_symptoms = set()
all_numerical = set()
all_labels = set()
datasets = []

# Load datasets
for file in dataset_files:
    path = os.path.join(dataset_folder, file)
    if not os.path.exists(path):
        logging.warning(f"{file} not found, skipping...")
        continue
    df = pd.read_csv(path)
    logging.debug(f"Loaded {file} with shape {df.shape}")
    datasets.append(df)

    # Assume non-numeric columns are symptoms or labels
    for col in df.columns:
        if df[col].dtype == 'object':
            all_symptoms.add(col)
        else:
            all_numerical.add(col)

# For simplicity, treat all object columns as symptoms and the last column as label
master_rows = []
for df in datasets:
    label_col = df.columns[-1]
    all_labels.add(label_col)
    numerical_cols = [col for col in df.columns if col not in all_symptoms and col != label_col]
    symptom_cols = [col for col in df.columns if col in all_symptoms and col != label_col]

    for idx, row in df.iterrows():
        row_dict = {}
        # Set all symptoms to 0
        for s in all_symptoms:
            row_dict[s] = 0
        # Mark present symptoms
        for s in symptom_cols:
            if str(row[s]).strip() != '' and str(row[s]).lower() not in ['no', '0', 'nan']:
                row_dict[s] = 1

        # Add numerical values, fill missing with None
        for n in all_numerical:
            row_dict[n] = row[n] if n in row else None

        # Add label
        row_dict["disease"] = row[label_col] if label_col in row else "Unknown"

        # Append as single-row DataFrame
        master_rows.append(pd.DataFrame({k:[v] for k,v in row_dict.items()}))

# Concatenate all rows
master_df = pd.concat(master_rows, ignore_index=True)
logging.debug(f"Master dataset shape: {master_df.shape}")

# Fill numerical missing values with NaN (optional)
for col in all_numerical:
    if col in master_df.columns:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

# Save master dataset
master_csv_path = os.path.join(dataset_folder, "master_dataset.csv")
master_df.to_csv(master_csv_path, index=False)
logging.debug(f"Master dataset saved to {master_csv_path}")
