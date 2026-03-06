import os
import pandas as pd
from typing import Tuple, Dict, Any

def read_sample(path: str, nrows: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    _, ext = os.path.splitext(path.lower())
    if ext in [".csv", ".tsv"]:
        df = pd.read_csv(path, nrows=nrows)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path, nrows=nrows)
    elif ext == ".json":
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path)
        df = df.head(nrows)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    meta = {
        "ext": ext,
        "rows_sampled": len(df),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    return df, meta
