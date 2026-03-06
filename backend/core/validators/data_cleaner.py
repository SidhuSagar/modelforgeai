import os
import pandas as pd
from typing import Dict, Any

def clean_dataset_simple(file_path: str, out_path: str, strategy: str = "dropna") -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in [".csv", ".tsv"]:
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    elif ext == ".json":
        try:
            df = pd.read_json(file_path, lines=True)
        except ValueError:
            df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type")

    actions = []
    before_rows = df.shape[0]

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates()
        actions.append(f"dropped_duplicates:{dup_count}")

    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        if strategy == "dropna":
            df = df.dropna()
            actions.append("dropped_rows_with_na")
        elif strategy == "ffill":
            df = df.fillna(method="ffill").fillna(method="bfill")
            actions.append("filled_na_ffill_bfill")
        elif strategy == "median":
            num_cols = df.select_dtypes(include=["number"]).columns
            for c in num_cols:
                df[c].fillna(df[c].median(), inplace=True)
            actions.append("filled_numeric_median")
        else:
            df = df.fillna(method="ffill").fillna(method="bfill")
            actions.append("filled_na_default")

    after_rows = df.shape[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    return {
        "out_path": out_path,
        "before_rows": int(before_rows),
        "after_rows": int(after_rows),
        "actions": actions
    }
