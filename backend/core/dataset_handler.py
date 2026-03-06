# core/dataset_handler.py
"""
Universal Dataset Loader + Preprocessor for Model Forge AI
----------------------------------------------------------
✅ Robust CSV reading (multi-encoding + separator sniff + fallback)
✅ Duplicate column handling
✅ Auto-detect message/text + label columns
✅ Label normalization for common variants
✅ Drops unlabeled & duplicate rows by default (best training quality)
✅ Preserves all valid text rows
✅ Saves cleaned dataset copy to /datasets/processed/
✅ Supports .csv, .yml/.yaml, .txt, .md
"""

import os
import csv
import yaml
from typing import Optional, Union, List, Dict, Any, Tuple
import pandas as pd

# -------------------- Directory setup --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
RAW_DIR = os.path.join(DATASET_DIR, "raw")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# -------------------- Logging helper --------------------
def _log(msg: str):
    print(msg)

# -------------------- Safe CSV reader --------------------
def _safe_read_csv(file_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252"]
    seps = [",", "\t", ";"]
    tried = []

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                # Handle duplicate column names
                if df.columns.duplicated().any():
                    cols = pd.Series(df.columns)
                    for dup in cols[cols.duplicated()].unique():
                        dup_indices = cols[cols == dup].index
                        for i, idx in enumerate(dup_indices[1:], start=1):
                            cols[idx] = f"{dup}_{i}"
                    df.columns = cols
                if df.shape[1] > 0 and len(df) > 0:
                    _log(f"✅ Loaded CSV (encoding={enc}, sep='{sep}', rows={len(df)}, cols={len(df.columns)})")
                    return df
                tried.append(f"{enc}|{sep}: no valid data")
            except Exception as e:
                tried.append(f"{enc}|{sep}: {e}")

    # Try csv.Sniffer
    try:
        with open(file_path, "rb") as bf:
            sample = bf.read(16384)
        text = sample.decode("utf-8", errors="replace")
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text)
        sep = dialect.delimiter
        df = pd.read_csv(file_path, encoding="utf-8", sep=sep, engine="python", on_bad_lines="skip")
        _log(f"✅ Loaded CSV using Sniffer (sep='{sep}', rows={len(df)}, cols={len(df.columns)})")
        return df
    except Exception as e:
        tried.append(f"sniffer: {e}")

    # Fallback parser
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        rows = [l.split(",") for l in lines]
        max_cols = max(len(r) for r in rows)
        padded = [r + [""] * (max_cols - len(r)) for r in rows]
        header = padded[0]
        if any(h.strip() == "" for h in header):
            header = [f"col_{i}" for i in range(max_cols)]
            data = padded
        else:
            data = padded[1:]
        df = pd.DataFrame(data, columns=header)
        _log(f"⚠️ Fallback parser used ({len(df)} rows, {len(df.columns)} cols)")
        return df
    except Exception as e:
        tried.append(f"fallback: {e}")
        raise ValueError(f"Failed to parse CSV: {tried}")

# -------------------- Save processed helper --------------------
def _save_processed(obj: Union[pd.DataFrame, dict, list], name: str) -> str:
    safe_name = name.replace("/", "_").replace("\\", "_")
    if isinstance(obj, pd.DataFrame):
        path = os.path.join(PROCESSED_DIR, f"{safe_name}.csv")
        obj.to_csv(path, index=False, encoding="utf-8")
    else:
        path = os.path.join(PROCESSED_DIR, f"{safe_name}.yml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(obj, f, allow_unicode=True)
    return path

# -------------------- Detection Helpers --------------------
_TEXT_CANDIDATES = ["text","message","review","content","body","tweet","sentence","comment","question","article","input"]
_LABEL_CANDIDATES = ["label","sentiment","target","class","category","rating","score","tag","output","response"]

def _detect_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    cols = list(df.columns)
    low = [str(c).lower().strip() for c in cols]
    text_col = None

    # Try direct/partial text match
    for t in _TEXT_CANDIDATES:
        if t in low:
            text_col = cols[low.index(t)]
            break
    if not text_col:
        for i, c in enumerate(low):
            if any(t in c for t in _TEXT_CANDIDATES):
                text_col = cols[i]; break
    if not text_col:
        obj_cols = [c for c in cols if pd.api.types.is_object_dtype(df[c])]
        if obj_cols:
            avg_len = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
            text_col = max(avg_len, key=avg_len.get)
        else:
            text_col = cols[0]

    # Find label column
    label_col = _detect_label_column(df, text_col)
    return text_col, label_col

def _detect_label_column(df: pd.DataFrame, text_col: str) -> Optional[str]:
    cols = list(df.columns)
    low = [str(c).lower().strip() for c in cols]
    for l in _LABEL_CANDIDATES:
        if l in low and cols[low.index(l)] != text_col:
            return cols[low.index(l)]
    # Partial matches
    for i, c in enumerate(low):
        if any(l in c for l in _LABEL_CANDIDATES) and cols[i] != text_col:
            return cols[i]
    # Unique count heuristic
    n = len(df)
    for c in cols:
        if c == text_col: continue
        try:
            u = df[c].nunique(dropna=True)
            if 2 <= u <= max(50, int(0.05*n)):
                return c
        except Exception:
            continue
    if len(cols) == 2:
        return [c for c in cols if c != text_col][0]
    return None

# -------------------- Label Normalization --------------------
_LABEL_MAP = {
    "pos":"positive","positive":"positive","1":"positive","yes":"positive","true":"positive",
    "neg":"negative","negative":"negative","0":"negative","no":"negative","false":"negative",
    "spam":"spam","ham":"ham","not spam":"ham",
    "neutral":"neutral","neu":"neutral","2":"neutral",
    "none":"","nan":"","n/a":"","na":"","null":""
}

def _normalize_label(v: Any) -> str:
    if v is None or pd.isna(v): return ""
    s = str(v).strip().strip('"').strip("'")
    if s == "": return ""
    k = s.lower()
    if k in _LABEL_MAP: return _LABEL_MAP[k]
    try:
        f = float(k)
        if f.is_integer() and str(int(f)) in _LABEL_MAP:
            return _LABEL_MAP[str(int(f))]
    except: pass
    return s[:60]

# -------------------- Cleaning --------------------
def _clean_classification_df(df: pd.DataFrame, keep_duplicates=False, drop_unlabeled=True, min_per_class=2) -> pd.DataFrame:
    df = df.copy()
    _log(f"📊 Starting cleaning: {len(df)} rows, {len(df.columns)} cols")
    text_col, label_col = _detect_columns(df)
    _log(f"🔍 Detected text='{text_col}', label='{label_col}'")

    df["message"] = df[text_col].astype(str).fillna("").str.strip()
    df["label_raw"] = df[label_col].astype(str).fillna("").str.strip() if label_col else ""
    df = df[df["message"].str.len() > 0].copy()

    df["label"] = df["label_raw"].apply(_normalize_label)
    if drop_unlabeled:
        before = len(df)
        df = df[df["label"].str.len() > 0]
        _log(f"🗑️ Dropped {before - len(df)} unlabeled rows")
    if not keep_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["message","label"])
        _log(f"🗑️ Dropped {before - len(df)} duplicates")

    if df["label"].nunique() < 2:
        _log(f"⚠️ Warning: Only {df['label'].nunique()} unique labels")

    counts = df["label"].value_counts()
    too_small = counts[counts < min_per_class]
    if not too_small.empty:
        to_add = []
        for lbl, cnt in too_small.items():
            rows = df[df["label"]==lbl]
            while len(rows)+len([x for x in to_add if x["label"]==lbl])<min_per_class:
                to_add.append(rows.iloc[0].to_dict())
        if to_add:
            df = pd.concat([df, pd.DataFrame(to_add)], ignore_index=True)
            _log(f"📈 Upsampled {len(to_add)} rows to reach {min_per_class}/class")

    _log(f"✅ Final dataset: {len(df)} rows, labels={df['label'].nunique()}")
    return df[["message","label"]]

# -------------------- Loaders --------------------
def load_custom_dataset(file_path: str, **kwargs):
    file_path = os.path.abspath(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = _safe_read_csv(file_path)
        cleaned = _clean_classification_df(df, **kwargs)
        out = _save_processed(cleaned, os.path.basename(file_path).replace(".csv","_clean"))
        _log(f"💾 Saved cleaned dataset: {out}")
        return cleaned
    elif ext in (".yml",".yaml"):
        with open(file_path,"r",encoding="utf-8") as f: return yaml.safe_load(f)
    elif ext in (".txt",".md"):
        with open(file_path,"r",encoding="utf-8",errors="replace") as f: return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_classification_dataset(name="sms_spam", **kwargs):
    f = os.path.join(DATASET_DIR,"classification","spam.csv")
    if not os.path.exists(f): f=os.path.join(RAW_DIR,"spam.csv")
    df = _safe_read_csv(f)
    if set(["v1","v2"]).issubset(df.columns):
        df = df[["v1","v2"]]; df.columns=["label","message"]
    cleaned = _clean_classification_df(df, **kwargs)
    _save_processed(cleaned, f"{name}_clean")
    return cleaned

def load_chatbot_dataset(name="faq"):
    f = os.path.join(DATASET_DIR,"chatbot",f"{name}.yml")
    if not os.path.exists(f): f=os.path.join(RAW_DIR,f"{name}.yml")
    with open(f,"r",encoding="utf-8") as fh:
        data=yaml.safe_load(fh)
    if isinstance(data,dict): data=data.get("faqs",[data])
    _save_processed(data,f"{name}_clean")
    return data

def load_knowledge_dataset():
    folder = os.path.join(DATASET_DIR,"knowledge")
    if not os.path.exists(folder): folder=os.path.join(RAW_DIR,"knowledge")
    return [os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.lower().endswith((".pdf",".txt",".md"))]

def load_dataset(task_type: str, dataset_name=None, custom_path=None, **kwargs):
    task_type=(task_type or "").lower()
    if custom_path:
        return load_custom_dataset(custom_path, **kwargs)
    if task_type=="classification": return load_classification_dataset(dataset_name or "sms_spam", **kwargs)
    if task_type=="chatbot": return load_chatbot_dataset(dataset_name or "faq")
    if task_type=="knowledge": return load_knowledge_dataset()
    raise ValueError(f"Unknown task type: {task_type}")
