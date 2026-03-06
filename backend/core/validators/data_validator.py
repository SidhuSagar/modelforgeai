import os
import pandas as pd
from typing import Dict, Any
from .inspector import read_sample

def _read_full(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    elif ext == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported: {ext}")

def validate_dataset(file_path: str, sample_rows: int = 5) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    try:
        sample_df, sample_meta = read_sample(file_path, nrows=sample_rows)
        report.update({"sample_meta": sample_meta})
    except Exception as e:
        return {"status": "error", "error": f"read_sample_failed: {str(e)}"}

    try:
        df = _read_full(file_path)
    except Exception as e:
        return {"status": "error", "error": f"read_full_failed: {str(e)}"}

    rows, cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    missing_per_column = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())

    text_candidates = [c for c in df.columns if df[c].dtype == object and df[c].map(lambda x: isinstance(x, str)).all()]
    label_candidates = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        name = c.lower()
        if nunique <= 50 or name in ("label", "target", "sentiment", "class"):
            label_candidates.append(c)

    report.update({
        "status": "ok",
        "rows": rows,
        "columns": cols,
        "missing_total": missing_total,
        "missing_per_column": {k: int(v) for k, v in missing_per_column.items()},
        "duplicate_rows": duplicate_rows,
        "text_candidates": text_candidates,
        "label_candidates": label_candidates
    })
    return report
