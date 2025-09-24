"""
End-to-end dataset unification and training functions
"""
import os
import re
import numpy as np
import pandas as pd
from data_utils import list_uploaded_files, read_csv_safe, detect_label_columns, map_labels
from symptoms import SymptomMapper
from predictor import SmartHealthPredictor

def unify_and_build(data_dir: str, symptom_csv: str = None):
    """
    Load CSV files from data_dir, harmonize column names, detect labels, and construct
    a unified DataFrame X and labels DataFrame y. Also returns a SymptomMapper instance.
    """
    files = list_uploaded_files(data_dir)
    dfs = []
    label_frames = []

    for p in files:
        try:
            df = read_csv_safe(p)
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]

        # detect and map labels
        found = detect_label_columns(df)
        labels = map_labels(df, found) if found else pd.DataFrame(index=df.index)

        # normalize common column names
        colmap = {}
        rename_pairs = {
            'bloodpressure': 'BloodPressure', 'blood_pressure': 'BloodPressure', 'bp': 'BloodPressure',
            'glucose': 'Glucose', 'bpm': 'heartRate', 'heartrate': 'heartRate', 'age': 'Age',
            'bmi': 'BMI', 'sex': 'Sex', 'gender': 'Sex', 'smoker': 'currentSmoker', 'cigsperday': 'cigsPerDay',
            'serumcreatinine': 'Sc', 'serum_creatinine': 'Sc', 'sc': 'Sc', 'serum_creatinine_mg_dl': 'Sc',
            'blood_urea': 'Bu', 'bloodurea': 'Bu', 'bu': 'Bu', 'hemoglobin': 'Hemo', 'hemoglobin_g_dL': 'Hemo',
            'cholesterol': 'totChol'
        }
        for c in df.columns:
            key = c.replace(' ', '').lower()
            if key in rename_pairs:
                colmap[c] = rename_pairs[key]
        df = df.rename(columns=colmap)

        if not labels.empty:
            label_frames.append(labels)
        dfs.append(df)

    if not dfs:
        raise ValueError('No CSV files were found in the data directory.')

    all_df = pd.concat(dfs, ignore_index=True, sort=False)

    if label_frames:
        labels_all = pd.concat(label_frames, ignore_index=True, sort=False)
        if len(labels_all) != len(all_df):
            labels_all = labels_all.reindex(all_df.index, fill_value=0)
    else:
        # default zero labels
        labels_all = pd.DataFrame(0, index=all_df.index, columns=['ckd','diabetes','heart'])

    # instantiate symptom mapper
    sm = SymptomMapper(symptom_csv if symptom_csv and os.path.exists(symptom_csv) else None)

    # if symptom mapper has known symptoms, ensure those columns exist in all_df (filled with NaN -> later imputed)
    if sm.symptom_lookup:
        for s in sm.symptom_lookup.keys():
            if s not in all_df.columns:
                all_df[s] = np.nan

    # replace common nil indicators with NaN
    all_df = all_df.replace(['?', 'na', 'nan', 'None', ''], np.nan)

    return all_df, labels_all, sm

def build_and_train(data_dir: str, save_path: str = None, symptom_csv: str = None):
    print('Reading and unifying datasets from', data_dir)
    X, y, sm = unify_and_build(data_dir, symptom_csv)

    # drop useless columns
    drop_candidates = [c for c in X.columns if X[c].nunique() <= 1 or re.search(r'id$|patient', c.lower())]
    X = X.drop(columns=drop_candidates, errors='ignore')

    shp = SmartHealthPredictor()
    shp.fit(X, y, symptom_mapper=sm)

    if save_path:
        shp.save(save_path)
        print('Saved trained pipeline to', save_path)
    return shp

