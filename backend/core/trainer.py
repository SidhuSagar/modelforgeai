"""
core/trainer.py
Unified Trainer for ModelForge AI (improved)

Features / Notes:
- Supports: classification, chatbot, knowledge
- Auto-detects large datasets by file size and uses incremental/streaming approaches
- Prefers existing <name>_train.csv and <name>_test.csv created by your preprocessor
- Produces model .pkl and metadata .json in models/<task_type>/
- Uses joblib compression when saving
"""
import os
import json
import time
import math
from typing import Dict, Any, Optional, Tuple, Set

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse as sp_sparse

# Optional: sentence-transformers for knowledge base embeddings
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False

# Constants
MODEL_DIR = "models"
LARGE_FILE_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB -> treat as "large"
CSV_CHUNKSIZE = 100_000  # default chunk size for reading large csv
JOBLIB_COMPRESSION = 3


# ----------------------------
# Utilities
# ----------------------------
def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _task_dir(task_type: str) -> str:
    d = os.path.join(MODEL_DIR, task_type)
    os.makedirs(d, exist_ok=True)
    return d


def _save_model(obj: Any, task_type: str, model_name: str) -> str:
    _ensure_dirs()
    path = os.path.join(_task_dir(task_type), f"{model_name}.pkl")
    joblib.dump(obj, path, compress=JOBLIB_COMPRESSION)
    size = os.path.getsize(path) if os.path.exists(path) else None
    return path


def _save_metadata(task_type: str, model_name: str, metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    meta = {
        "model_name": model_name,
        "task_type": task_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    if extra:
        meta.update(extra)
    path = os.path.join(_task_dir(task_type), f"{model_name}_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def _is_large_file(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        return size >= LARGE_FILE_THRESHOLD_BYTES
    except Exception:
        return False


def _try_find_splits(cleaned_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    If cleaned_path ends with *_cleaned.csv, check for *_train.csv and *_test.csv.
    Returns (train_path, test_path) or (None, None).
    """
    base, ext = os.path.splitext(cleaned_path)
    if base.endswith("_cleaned"):
        prefix = base[:-len("_cleaned")]
    else:
        prefix = base
    train_path = prefix + "_train.csv"
    test_path = prefix + "_test.csv"
    return (train_path if os.path.exists(train_path) else None,
            test_path if os.path.exists(test_path) else None)


def _unique_labels_from_csv(path: str, label_col: str = "label", chunksize: int = CSV_CHUNKSIZE) -> Set[str]:
    """
    One-pass to gather unique labels from a possibly huge CSV using chunks.
    """
    uniq = set()
    for chunk in pd.read_csv(path, usecols=[label_col], chunksize=chunksize, dtype=str, low_memory=True):
        uniq.update([v for v in chunk[label_col].dropna().unique()])
    return uniq


# ----------------------------
# Classification training
# ----------------------------
def train_classification_model(
    dataset_path: str,
    model_name: str = "classification_model",
    epochs: int = 10,
    text_col: str = "message",
    label_col: str = "label",
) -> Dict[str, Any]:
    """
    Safe classification training for ModelForge AI.
    Handles rare classes, bad splits, and fallback vectorizers automatically.
    """
    print(f"[classification] dataset: {dataset_path}")
    train_path, test_path = _try_find_splits(dataset_path)

    large = _is_large_file(dataset_path) or (_is_large_file(train_path) if train_path else False)
    print(f" - large dataset: {large}")

    metrics = {}

    # ---------- Large dataset (streaming SGDClassifier) ----------
    if large:
        print(" - Using streaming incremental training (HashingVectorizer + SGDClassifier)")
        vectorizer = HashingVectorizer(n_features=2**20, alternate_sign=False, norm=None)
        clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None)
        classes = None

        # Fetch labels for incremental learning
        if train_path:
            classes = sorted(list(_unique_labels_from_csv(train_path, label_col)))
        else:
            classes = sorted(list(_unique_labels_from_csv(dataset_path, label_col)))

        if len(classes) < 2:
            raise ValueError(f"Dataset has <2 classes ({classes}); cannot train classifier.")

        def _stream_csv_and_partial_fit(path):
            nonlocal clf
            for chunk in pd.read_csv(path, usecols=[text_col, label_col], chunksize=CSV_CHUNKSIZE, dtype=str):
                chunk = chunk.dropna(subset=[text_col, label_col])
                if chunk.empty:
                    continue
                texts = chunk[text_col].astype(str).tolist()
                labels = chunk[label_col].astype(str).tolist()
                X = vectorizer.transform(texts)
                if not hasattr(clf, "classes_"):
                    clf.partial_fit(X, labels, classes=classes)
                else:
                    clf.partial_fit(X, labels)

        _stream_csv_and_partial_fit(train_path or dataset_path)
        model_obj = {"vectorizer": vectorizer, "classifier": clf, "classes": classes}
        model_path = _save_model(model_obj, "classification", model_name)
        meta_path = _save_metadata("classification", model_name, metrics, {"classes": classes})
        return {
            "model_name": model_name,
            "task_type": "classification",
            "model_path": model_path,
            "meta_path": meta_path,
            "metrics": metrics,
        }

    # ---------- Regular TF-IDF + LogisticRegression ----------
    print(" - Using TF-IDF + LogisticRegression (safe mode)")
    if train_path and test_path:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    else:
        df = pd.read_csv(dataset_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError("Dataset must contain columns: 'message' and 'label'.")

        # ⚙️ Attempt train/test split safely
        try:
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
        except ValueError as e:
            print(f"⚠️ Stratified split failed ({e}). Retrying with test_size=0.25 (non-stratified).")
            try:
                df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
            except Exception as e2:
                print(f"⚠️ Retry failed ({e2}). Using full dataset as training only.")
                df_train = df.copy()
                df_test = df.sample(n=min(50, len(df)), random_state=42) if len(df) > 50 else df.copy()

    # Transform and train
    X_train = df_train[text_col].astype(str)
    y_train = df_train[label_col].astype(str)
    X_test = df_test[text_col].astype(str)
    y_test = df_test[label_col].astype(str)

    if len(set(y_train)) < 2:
        raise ValueError(f"Training data has only one class ({set(y_train)}); cannot train model.")

    vectorizer = TfidfVectorizer(max_features=20000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=max(100, epochs))
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    metrics["accuracy"] = round(float(acc), 4)
    print(f" - accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    model_obj = {"vectorizer": vectorizer, "classifier": clf}
    model_path = _save_model(model_obj, "classification", model_name)
    meta_path = _save_metadata("classification", model_name, metrics)

    return {
        "model_name": model_name,
        "task_type": "classification",
        "model_path": model_path,
        "meta_path": meta_path,
        "metrics": metrics,
    }

# ----------------------------
# Chatbot training (retrieval-based)
# ----------------------------
def train_chatbot_model(
    dataset_path: str,
    model_name: str = "chatbot_model",
    question_col: str = "question",
    answer_col: str = "answer",
) -> Dict[str, Any]:
    """
    Builds a retrieval-based FAQ chatbot:
    - Stores vectorizer, question list, answers and question matrix.
    - For large datasets, uses HashingVectorizer and saves sparse matrix.
    """
    print(f"[chatbot] dataset: {dataset_path}")
    # Load dataset (prefer train if present, but chatbot uses full faq)
    train_path, _ = _try_find_splits(dataset_path)
    path = train_path or dataset_path
    large = _is_large_file(path)
    print(f" - large dataset: {large}")

    if large:
        # HashingVectorizer + incremental transform & save sparse matrix to disk
        print(" - Using HashingVectorizer for large dataset (sparse matrix will be saved).")
        vectorizer = HashingVectorizer(n_features=2**20, alternate_sign=False, norm='l2')
        questions = []
        answers = []
        rows = []
        # We'll build a CSR matrix chunk-by-chunk and stack on disk using save_npz chunks
        matrices = []
        for chunk in pd.read_csv(path, usecols=[question_col, answer_col], chunksize=CSV_CHUNKSIZE, dtype=str, low_memory=True):
            chunk = chunk.dropna(subset=[question_col, answer_col])
            q_list = chunk[question_col].astype(str).tolist()
            a_list = chunk[answer_col].astype(str).tolist()
            X_chunk = vectorizer.transform(q_list)
            matrices.append(X_chunk)
            questions.extend(q_list)
            answers.extend(a_list)

        # vertically stack sparse matrices
        q_matrix = sp_sparse.vstack(matrices).tocsr()
        # Save sparse matrix separately to avoid storing huge object in memory model
        matrix_path = os.path.join(_task_dir("chatbot"), f"{model_name}_qmatrix.npz")
        sp_sparse.save_npz(matrix_path, q_matrix)

        model_obj = {
            "vectorizer_type": "hashing",
            "vectorizer": vectorizer,
            "questions": questions,
            "answers": answers,
            "matrix_path": matrix_path,
        }
        model_path = _save_model(model_obj, "chatbot", model_name)
        meta_path = _save_metadata("chatbot", model_name, {"n_samples": len(questions), "matrix_path": matrix_path})
        print(f" - saved qmatrix: {matrix_path}")
        return {
            "model_name": model_name,
            "task_type": "chatbot",
            "model_path": model_path,
            "meta_path": meta_path,
            "n_samples": len(questions),
        }

    # Non-large: fit TfidfVectorizer on all questions
    df = pd.read_csv(path)
    df = df.dropna(subset=[question_col, answer_col])
    questions = df[question_col].astype(str).tolist()
    answers = df[answer_col].astype(str).tolist()

    vectorizer = TfidfVectorizer(max_features=10000)
    q_matrix = vectorizer.fit_transform(questions)

    model_obj = {
        "vectorizer_type": "tfidf",
        "vectorizer": vectorizer,
        "questions": questions,
        "answers": answers,
        "matrix": q_matrix,  # sparse matrix inside model_obj; joblib handles sparse
    }
    model_path = _save_model(model_obj, "chatbot", model_name)
    meta_path = _save_metadata("chatbot", model_name, {"n_samples": len(questions)})
    print(" - Chatbot model saved (TF-IDF).")
    return {
        "model_name": model_name,
        "task_type": "chatbot",
        "model_path": model_path,
        "meta_path": meta_path,
        "n_samples": len(questions),
    }


# ----------------------------
# Knowledge base training (embeddings)
# ----------------------------
def train_knowledge_model(
    dataset_path: str,
    model_name: str = "knowledge_model",
    doc_col: str = "document",
    batch_size: int = 4096,
) -> Dict[str, Any]:
    """
    Produces document embeddings. If sentence-transformers is available it will be used;
    otherwise falls back to TF-IDF document vectors.
    """
    print(f"[knowledge] dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if doc_col not in df.columns:
        raise ValueError("Knowledge dataset must have a 'document' column.")

    docs = df[doc_col].astype(str).tolist()
    n = len(docs)
    print(f" - n_documents: {n}")

    if _HAS_SENTENCE_TRANSFORMERS:
        print(" - Using sentence-transformers to build dense embeddings.")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # encode in batches (streaming)
        embeddings = encoder.encode(docs, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        model_obj = {"encoder_name": "all-MiniLM-L6-v2", "encoder": None, "documents": docs, "embeddings": embeddings}
        # We store encoder_name but NOT the encoder object to keep pkl small; user can re-load encoder by name
        model_path = _save_model(model_obj, "knowledge", model_name)
        meta_path = _save_metadata("knowledge", model_name, {"n_docs": n, "emb_shape": embeddings.shape})
        print(" - embeddings computed & saved.")
        return {
            "model_name": model_name,
            "task_type": "knowledge",
            "model_path": model_path,
            "meta_path": meta_path,
            "n_docs": n,
            "emb_shape": embeddings.shape,
        }

    # Fallback: TF-IDF / sparse vectors for docs
    print(" - sentence-transformers not installed; falling back to TF-IDF vectors.")
    vectorizer = TfidfVectorizer(max_features=20000)
    doc_matrix = vectorizer.fit_transform(docs)  # sparse
    model_obj = {"vectorizer": vectorizer, "documents": docs, "doc_matrix": doc_matrix}
    model_path = _save_model(model_obj, "knowledge", model_name)
    meta_path = _save_metadata("knowledge", model_name, {"n_docs": n, "matrix_shape": doc_matrix.shape})
    return {
        "model_name": model_name,
        "task_type": "knowledge",
        "model_path": model_path,
        "meta_path": meta_path,
        "n_docs": n,
        "matrix_shape": doc_matrix.shape,
    }


# ----------------------------
# Unified entrypoint
# ----------------------------
def train_model(task_type: str, dataset_path: str, model_name: Optional[str], epochs: int = 10) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model_name = model_name or f"{task_type}_model_{int(time.time())}"
    if task_type == "classification":
        return train_classification_model(dataset_path, model_name, epochs=epochs)
    elif task_type == "chatbot":
        return train_chatbot_model(dataset_path, model_name)
    elif task_type == "knowledge":
        return train_knowledge_model(dataset_path, model_name)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


# ----------------------------
# CLI for quick testing
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified trainer for ModelForge AI")
    parser.add_argument("task_type", choices=["classification", "chatbot", "knowledge"])
    parser.add_argument("dataset_path", help="path to preprocessed/cleaned CSV (or train/test variants)")
    parser.add_argument("--model-name", help="name for model artifact", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    result = train_model(args.task_type, args.dataset_path, args.model_name, args.epochs)
    print(json.dumps(result, indent=2))
