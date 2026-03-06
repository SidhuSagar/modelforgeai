"""
core/preprocessor.py

Robust streaming preprocessor for large datasets.

Features:
- Supports CSV (chunked), JSONL, Parquet (if pyarrow installed), and plain text.
- Cleans basic text (lowercase, strip, collapse whitespace, remove control chars).
- Deterministic train/test split using hashing (no need to load whole dataset).
- Writes processed outputs to datasets/processed/:
    - <output_name>_cleaned.csv       (all cleaned rows)
    - <output_name>_train.csv         (train split)
    - <output_name>_test.csv          (test split)
- Safe for very large files via pandas chunksize or line-by-line processing.
- Returns a dict with file paths and counts.

Usage:
    from core.preprocessor import preprocess_data
    info = preprocess_data(
        dataset_path="datasets/raw/sms_spam.csv",
        output_name="sms_spam",
        text_col="text",
        label_col="label",
        test_size=0.2,
        chunksize=100000
    )
"""
import os
import csv
import json
import hashlib
import math
from typing import Optional, Dict, Tuple, Iterable

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas is required for core.preprocessor. Install with `pip install pandas`.") from e

# Try optional dependency for parquet support
_HAS_PYARROW = False
try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

# Default processed dataset directory
PROCESSED_DIR = os.path.join("datasets", "processed")
RAW_DIR = os.path.join("datasets", "raw")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def _safe_text_clean(s: str) -> str:
    """Basic text cleaning suitable for NLP pipelines."""
    if s is None:
        return ""
    try:
        # Ensure string
        s = str(s)
    except Exception:
        return ""

    # Normalize whitespace and remove most control/non-printable chars
    # Lowercase and strip edges
    s = s.replace("\r", " ").replace("\n", " ").strip()
    # collapse multiple spaces/tabs
    s = " ".join(s.split())
    s = s.lower()
    # optionally strip weird non-printable
    s = "".join(ch for ch in s if ch.isprintable())
    return s


def _deterministic_split_key(value: str, salt: Optional[str] = None) -> float:
    """
    Hash a string deterministically and return a float in [0,1).
    Use this for assign-to-train/test without loading entire dataset.
    """
    if value is None:
        value = ""
    if salt:
        value = f"{salt}|{value}"
    h = hashlib.md5(value.encode("utf-8")).hexdigest()
    # take first 12 hex digits -> convert to int
    prefix = h[:12]
    intval = int(prefix, 16)
    maxval = float(int("f" * 12, 16))
    return intval / maxval


def _guess_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristically guess text & label columns from a DataFrame.
    """
    text_candidates = ["text", "message", "content", "sentence", "utterance", "query"]
    label_candidates = ["label", "target", "class", "intent", "category"]

    cols = [c.lower() for c in df.columns]
    text_col = None
    label_col = None

    for cand in text_candidates:
        if cand in cols:
            text_col = df.columns[cols.index(cand)]
            break

    for cand in label_candidates:
        if cand in cols:
            label_col = df.columns[cols.index(cand)]
            break

    return text_col, label_col


# ----------------------------
# Core preprocessing function
# ----------------------------
def preprocess_data(
    dataset_path: str,
    output_name: Optional[str] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    test_size: float = 0.2,
    salt: Optional[str] = None,
    chunksize: int = 100_000,
    encoding: str = "utf-8",
    save_cleaned: bool = True,
    save_splits: bool = True,
    min_text_length: int = 1,
) -> Dict:
    """
    Preprocess a dataset in a streaming/chunked manner and write cleaned outputs.

    Parameters
    ----------
    dataset_path : str
        Path to the input dataset (csv, jsonl, parquet, or txt).
    output_name : Optional[str]
        Base name for output files. If None, derived from input filename.
    text_col : Optional[str]
        Column name containing text. If not provided, will be guessed (CSV/Parquet).
        For jsonl you should provide this for best results.
    label_col : Optional[str]
        Column name containing label (if any). If not found, label column will be empty.
    test_size : float
        Fraction assigned to test set (0-1). Deterministic hashing used.
    salt : Optional[str]
        Salt used in hashing to vary splits reproducibly across runs.
    chunksize : int
        Number of rows to read per chunk for CSV / Parquet. Larger is faster but uses more memory.
    encoding : str
        File encoding for CSV/JSONL/txt.
    save_cleaned : bool
        Whether to save a combined cleaned CSV (all rows).
    save_splits : bool
        Whether to save train/test CSV splits separately.
    min_text_length : int
        Drop rows with text shorter than this after cleaning.

    Returns
    -------
    dict
        {
            "cleaned_path": ... (or None),
            "train_path": ... (or None),
            "test_path": ... (or None),
            "n_rows_total": int,
            "n_train": int,
            "n_test": int
        }
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Input dataset not found: {dataset_path}")

    if output_name is None:
        base = os.path.basename(dataset_path)
        output_name = os.path.splitext(base)[0]

    cleaned_path = os.path.join(PROCESSED_DIR, f"{output_name}_cleaned.csv") if save_cleaned else None
    train_path = os.path.join(PROCESSED_DIR, f"{output_name}_train.csv") if save_splits else None
    test_path = os.path.join(PROCESSED_DIR, f"{output_name}_test.csv") if save_splits else None

    # Remove existing files to avoid appending to old data unintentionally
    for p in [cleaned_path, train_path, test_path]:
        if p and os.path.exists(p):
            os.remove(p)

    n_total = 0
    n_train = 0
    n_test = 0
    is_first_chunk = True

    # Very small helper to append a dataframe to a CSV with correct header handling
    def _append_df_to_csv(df: pd.DataFrame, path: str):
        mode = "a"
        header = not os.path.exists(path)
        df.to_csv(path, index=False, encoding=encoding, mode=mode, header=header)

    # Handler for CSV files (chunked)
    if dataset_path.lower().endswith(".csv"):
        # Use pandas read_csv in chunks
        for chunk in pd.read_csv(dataset_path, chunksize=chunksize, encoding=encoding, low_memory=False):
            # On the first chunk, if text_col/label_col not provided, try to guess
            if is_first_chunk:
                if text_col is None or label_col is None:
                    guessed_text, guessed_label = _guess_columns(chunk)
                    if text_col is None:
                        text_col = guessed_text
                    if label_col is None:
                        label_col = guessed_label
                is_first_chunk = False

            # If we still couldn't find text_col, raise
            if text_col is None or text_col not in chunk.columns:
                raise ValueError(
                    "text_col could not be determined. Provide text_col explicitly for CSV or ensure a column named 'text' or similar exists."
                )

            # Keep only relevant columns (text + label if present)
            cols_to_keep = [text_col]
            if label_col and label_col in chunk.columns:
                cols_to_keep.append(label_col)
            df_chunk = chunk[cols_to_keep].copy()

            # Clean text column
            df_chunk[text_col] = df_chunk[text_col].apply(_safe_text_clean)

            # Drop very short texts or empty
            df_chunk = df_chunk[df_chunk[text_col].str.len() >= min_text_length]

            # Determine split per row using deterministic hash on text (or on label+text if you want)
            # For rows containing NaN text after cleaning, they were dropped above
            split_vals = df_chunk[text_col].apply(lambda s: _deterministic_split_key(s, salt))
            is_test = split_vals < test_size  # choose threshold for test set

            # Write cleaned combined file
            if save_cleaned:
                _append_df_to_csv(df_chunk, cleaned_path)

            # Write train/test
            if save_splits:
                train_df = df_chunk[~is_test]
                test_df = df_chunk[is_test]
                if not train_df.empty:
                    _append_df_to_csv(train_df, train_path)
                    n_train += len(train_df)
                if not test_df.empty:
                    _append_df_to_csv(test_df, test_path)
                    n_test += len(test_df)

            n_total += len(df_chunk)

    # Handler for JSONL (each line a JSON object)
    elif dataset_path.lower().endswith(".jsonl") or dataset_path.lower().endswith(".ndjson"):
        # If text_col not provided, we will attempt to infer from first line
        with open(dataset_path, "r", encoding=encoding) as f:
            header_checked = False
            header_inferred_text = None
            # We'll build small batches to write more efficiently
            batch = []
            batch_size = 20000
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not header_checked:
                    if text_col is None:
                        # try to guess text field name
                        for cand in ["text", "message", "content", "sentence"]:
                            if cand in obj:
                                header_inferred_text = cand
                                break
                        text_col = text_col or header_inferred_text
                    if label_col is None:
                        for cand in ["label", "class", "target"]:
                            if cand in obj:
                                label_col = label_col or cand
                    header_checked = True

                # Build a small dict
                text_val = obj.get(text_col) if text_col else None
                label_val = obj.get(label_col) if label_col else None
                text_clean = _safe_text_clean(text_val)
                if len(text_clean) < min_text_length:
                    continue
                batch.append({text_col: text_clean, label_col: label_val} if label_col else {text_col: text_clean})

                if len(batch) >= batch_size:
                    df_batch = pd.DataFrame(batch)
                    split_vals = df_batch[text_col].apply(lambda s: _deterministic_split_key(s, salt))
                    is_test = split_vals < test_size

                    if save_cleaned:
                        _append_df_to_csv(df_batch, cleaned_path)
                    if save_splits:
                        train_df = df_batch[~is_test]
                        test_df = df_batch[is_test]
                        if not train_df.empty:
                            _append_df_to_csv(train_df, train_path)
                            n_train += len(train_df)
                        if not test_df.empty:
                            _append_df_to_csv(test_df, test_path)
                            n_test += len(test_df)
                    n_total += len(df_batch)
                    batch = []

            # leftover
            if batch:
                df_batch = pd.DataFrame(batch)
                split_vals = df_batch[text_col].apply(lambda s: _deterministic_split_key(s, salt))
                is_test = split_vals < test_size
                if save_cleaned:
                    _append_df_to_csv(df_batch, cleaned_path)
                if save_splits:
                    train_df = df_batch[~is_test]
                    test_df = df_batch[is_test]
                    if not train_df.empty:
                        _append_df_to_csv(train_df, train_path)
                        n_train += len(train_df)
                    if not test_df.empty:
                        _append_df_to_csv(test_df, test_path)
                        n_test += len(test_df)
                n_total += len(df_batch)

    # Handler for Parquet
    elif dataset_path.lower().endswith(".parquet"):
        if not _HAS_PYARROW:
            raise RuntimeError("pyarrow / parquet support not available. Install pyarrow to preprocess parquet files.")
        # pd.read_parquet typically loads whole file; some parquet stores might be large but often manageable
        df = pd.read_parquet(dataset_path)
        if text_col is None or label_col is None:
            guessed_text, guessed_label = _guess_columns(df)
            text_col = text_col or guessed_text
            label_col = label_col or guessed_label
        if text_col is None or text_col not in df.columns:
            raise ValueError("text_col could not be determined for parquet file.")
        df[text_col] = df[text_col].apply(_safe_text_clean)
        df = df[df[text_col].str.len() >= min_text_length]
        # Splitting deterministic using hash
        splits = df[text_col].apply(lambda s: _deterministic_split_key(s, salt))
        is_test = splits < test_size
        if save_cleaned:
            df.to_csv(cleaned_path, index=False, encoding=encoding)
        if save_splits:
            df[~is_test].to_csv(train_path, index=False, encoding=encoding)
            df[is_test].to_csv(test_path, index=False, encoding=encoding)
            n_train = int((~is_test).sum())
            n_test = int(is_test.sum())
        n_total = len(df)

    # Handler for plain text (one sample per line)
    elif dataset_path.lower().endswith(".txt"):
        batch = []
        batch_size = 20000
        text_col_name = text_col or "text"
        with open(dataset_path, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                cleaned = _safe_text_clean(raw)
                if len(cleaned) < min_text_length:
                    continue
                batch.append({text_col_name: cleaned})
                if len(batch) >= batch_size:
                    df_batch = pd.DataFrame(batch)
                    split_vals = df_batch[text_col_name].apply(lambda s: _deterministic_split_key(s, salt))
                    is_test = split_vals < test_size
                    if save_cleaned:
                        _append_df_to_csv(df_batch, cleaned_path)
                    if save_splits:
                        train_df = df_batch[~is_test]
                        test_df = df_batch[is_test]
                        if not train_df.empty:
                            _append_df_to_csv(train_df, train_path)
                            n_train += len(train_df)
                        if not test_df.empty:
                            _append_df_to_csv(test_df, test_path)
                            n_test += len(test_df)
                    n_total += len(df_batch)
                    batch = []
            if batch:
                df_batch = pd.DataFrame(batch)
                split_vals = df_batch[text_col_name].apply(lambda s: _deterministic_split_key(s, salt))
                is_test = split_vals < test_size
                if save_cleaned:
                    _append_df_to_csv(df_batch, cleaned_path)
                if save_splits:
                    train_df = df_batch[~is_test]
                    test_df = df_batch[is_test]
                    if not train_df.empty:
                        _append_df_to_csv(train_df, train_path)
                        n_train += len(train_df)
                    if not test_df.empty:
                        _append_df_to_csv(test_df, test_path)
                        n_test += len(test_df)
                n_total += len(df_batch)

    else:
        raise ValueError("Unsupported file type. Supported: .csv, .jsonl/.ndjson, .parquet, .txt")

    # If splits were not explicitly saved but we did clean-only, we can optionally create counts by quick pass
    # (But we already counted n_total/n_train/n_test when we wrote splits.)

    result = {
        "cleaned_path": cleaned_path,
        "train_path": train_path,
        "test_path": test_path,
        "n_rows_total": int(n_total),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "output_name": output_name,
    }

    print(f"✅ Preprocessing finished. Total rows processed: {n_total}")
    if save_splits:
        print(f"    Train rows: {n_train} | Test rows: {n_test}")
    if cleaned_path:
        print(f"    Cleaned file: {cleaned_path}")
    if train_path:
        print(f"    Train file : {train_path}")
    if test_path:
        print(f"    Test file  : {test_path}")

    return result


# ----------------------------
# CLI convenience
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream / chunk preprocess a large dataset.")
    parser.add_argument("dataset_path", help="Input file path (.csv, .jsonl, .parquet, .txt)")
    parser.add_argument("--output-name", help="Base output name", default=None)
    parser.add_argument("--text-col", help="Text column name", default=None)
    parser.add_argument("--label-col", help="Label column name", default=None)
    parser.add_argument("--test-size", help="Fraction for test split", type=float, default=0.2)
    parser.add_argument("--chunksize", help="Rows per chunk for csv", type=int, default=100000)
    parser.add_argument("--salt", help="Salt for deterministic split", default=None)

    args = parser.parse_args()
    info = preprocess_data(
        dataset_path=args.dataset_path,
        output_name=args.output_name,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
        chunksize=args.chunksize,
        salt=args.salt,
    )
    print(json.dumps(info, indent=2))
