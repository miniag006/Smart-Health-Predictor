"""
Data utilities: safe CSV reading, file listing, label detection & mapping
"""
import os
import pandas as pd
import numpy as np
from typing import List

DEFAULT_LABELS = ['ckd', 'diabetes', 'heart']

def read_csv_safe(path: str) -> pd.DataFrame:
    """Try common encodings to read CSV robustly."""
    for enc in (None, 'latin-1', 'utf-8'):
        try:
            if enc:
                return pd.read_csv(path, encoding=enc)
            else:
                return pd.read_csv(path)
        except Exception:
            continue
    raise ValueError(f"Unable to read CSV: {path}")

def list_uploaded_files(data_dir: str) -> List[str]:
    """Return sorted list of CSV files in a directory."""
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    return sorted(files)

def detect_label_columns(df: pd.DataFrame) -> List[str]:
    """Detect likely label columns by name patterns."""
    found = []
    for lab in DEFAULT_LABELS:
        for c in df.columns:
            if lab in c.lower():
                found.append(c)
                break
    # fallback label names
    if not found:
        for candidate in ['class', 'outcome', 'tenyearchd', 'disease', 'target', 'label']:
            for c in df.columns:
                if candidate == c.lower():
                    found.append(c)
    return found

def map_labels(df: pd.DataFrame, found_labels: List[str]) -> pd.DataFrame:
    """
    Map detected label columns into a DataFrame with binary columns: ckd, diabetes, heart.
    """
    target = pd.DataFrame(index=df.index)
    for lab in DEFAULT_LABELS:
        target[lab] = 0

    for c in found_labels:
        low = c.lower()
        vals = df[c]
        mapped = None
        if 'ckd' in low or 'kidney' in low:
            mapped = 'ckd'
        elif 'diab' in low or 'glucose' in low or 'insulin' in low:
            mapped = 'diabetes'
        elif 'heart' in low or 'chd' in low:
            mapped = 'heart'
        else:
            unique_vals = vals.dropna().astype(str).str.lower().unique()
            if any('kidney' in s or 'ckd' in s for s in unique_vals):
                mapped = 'ckd'
            elif any('diab' in s or 'dm' in s for s in unique_vals):
                mapped = 'diabetes'
            elif any('heart' in s or 'chd' in s or 'card' in s for s in unique_vals):
                mapped = 'heart'

        if mapped:
            try:
                target[mapped] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
            except Exception:
                s = df[c].astype(str).str.lower()
                target[mapped] = s.apply(
                    lambda x: 1 if any(k in x for k in ['yes','1','positive','ckd','diabetes','heart','disease','dm','chd']) else 0
                )

    return target
