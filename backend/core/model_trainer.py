# core/model_trainer.py
"""
Enhanced trainer utilities (classification, chatbot index, knowledge embeddings).

Key functions:
 - train_classification(dataset, model_name=None, model_type="auto", test_size=0.2, random_state=42, hyperparams=None)
 - train_llm_classification(...)  (scaffold)
 - build_chatbot_index(dataset_name="faq", force=False)
 - build_knowledge_embeddings(force=False)
 - ensure_trained_models()
 - load_trained_model(model_path)
"""
from typing import Optional, Dict, Any, Tuple, List
import os
import time
import joblib
import pickle
import numpy as np

# optional libs (import failure is handled)
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTEVAL_AVAILABLE = True
except Exception:
    SENTEVAL_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _ensure_sklearn():
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn and pandas required. Install: pip install scikit-learn pandas joblib")


def _timestamp() -> int:
    return int(time.time())


def _get_latest_model_file(task_subdir: str, ext_filters: Tuple[str, ...] = (".joblib", ".pkl")) -> Optional[str]:
    dirpath = os.path.join(MODELS_DIR, task_subdir)
    if not os.path.exists(dirpath):
        return None
    candidates = []
    for fname in os.listdir(dirpath):
        if any(fname.endswith(ext) for ext in ext_filters) or os.path.isdir(os.path.join(dirpath, fname)):
            candidates.append(os.path.join(dirpath, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _auto_select_algorithm(X: List[str], y: List[str], candidate_algos=None) -> Tuple[str, Any]:
    _ensure_sklearn()
    if candidate_algos is None:
        candidate_algos = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "SVM": SVC(probability=True)
        }
    results = {}
    for name, estimator in candidate_algos.items():
        try:
            pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", estimator)])
            scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")
            results[name] = (scores.mean(), pipe)
        except Exception:
            results[name] = (0.0, None)
    best_name = max(results.items(), key=lambda kv: kv[1][0])[0]
    best_score, best_pipe = results[best_name]
    return best_name, best_pipe


def train_classification(
    dataset,  # DataFrame or path
    model_name: Optional[str] = None,
    model_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    hyperparams: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a text classification model (sklearn pipeline) or route to LLM scaffold.

    Returns metadata dict with keys including:
    - task_type, model_type, model_name, model_path, trained_at, accuracy, report, n_samples
    """
    _ensure_sklearn()

    import pandas as pd
    # load dataframe if path
    if isinstance(dataset, str):
        df = pd.read_csv(dataset)
    else:
        df = dataset

    if not isinstance(df, pd.DataFrame):
        raise ValueError("dataset must be pandas DataFrame or CSV path")

    if "message" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected 'message' and 'label' columns in dataset")

    X = df["message"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    # --- Safe data split ---
    num_classes = len(set(y))
    min_required = num_classes * 2  # ensure at least 2 samples per class

    if len(X) < min_required:
        # dataset too small — use full data for training (no proper test split)
        print(f"⚠️ Dataset very small ({len(X)} samples, {num_classes} classes). Using full data for training.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        # auto-adjust test_size for very small datasets so stratify works
        effective_test_size = test_size
        if len(X) * test_size < num_classes:
            # ensure at least one sample per class in test set where possible
            effective_test_size = max(num_classes / len(X), 0.1)
            print(f"⚙️ Adjusted test size to {effective_test_size:.2f} for small dataset.")
        try:
            stratify = y if num_classes > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=effective_test_size, random_state=random_state, stratify=stratify
            )
        except ValueError as e:
            # fallback to simple split (non-stratified) to avoid crash
            print(f"⚠️ Stratified split failed: {e}. Falling back to simple 80/20 split.")
            split_point = max(1, int(0.8 * len(X)))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]

    chosen = None
    pipeline = None
    mt = str(model_type).lower()
    if mt == "auto":
        # for auto select we need training data — safe because X_train may equal X
        chosen, pipeline = _auto_select_algorithm(X_train, y_train)
    elif mt in ("logisticregression", "logistic", "logistic_regression"):
        chosen = "LogisticRegression"
        pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", LogisticRegression(max_iter=1000))])
    elif mt in ("randomforest", "rf", "random_forest"):
        chosen = "RandomForest"
        pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", RandomForestClassifier(n_estimators=100))])
    elif mt in ("svm", "svc"):
        chosen = "SVM"
        pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", SVC(probability=True))])
    elif mt == "llm":
        return train_llm_classification(dataset=df, model_name=model_name, hyperparams=hyperparams)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if hyperparams and isinstance(hyperparams, dict):
        try:
            est = pipeline.named_steps.get("clf")
            if est:
                est.set_params(**hyperparams)
        except Exception:
            pass

    pipeline.fit(X_train, y_train)

    # If X_test is the same as X_train (tiny dataset case) this simply evaluates on training data.
    preds = pipeline.predict(X_test)
    try:
        acc = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, output_dict=True)
    except Exception:
        acc = 0.0
        report = {}

    timestamp = _timestamp()
    model_name = model_name or f"{chosen}_{timestamp}"
    model_dir = os.path.join(MODELS_DIR, "classification")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(pipeline, model_path)

    # legacy pickle (vectorizer, clf) for older inference code compatibility
    legacy_pickle = None
    try:
        vec = pipeline.named_steps.get("tfidf")
        clf = pipeline.named_steps.get("clf")
        legacy_pickle = os.path.join(model_dir, f"{model_name}.pkl")
        with open(legacy_pickle, "wb") as f:
            pickle.dump((vec, clf), f)
    except Exception:
        legacy_pickle = None

    metadata = {
        "task_type": "classification",
        "model_type": chosen,
        "model_name": model_name,
        "model_path": model_path,
        "legacy_pickle": legacy_pickle,
        "accuracy": acc,
        "report": report,
        "n_samples": len(df),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
    }
    return metadata


def train_llm_classification(dataset, model_name: Optional[str] = None, hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Minimal scaffold for LLM-based classifier. Saves pretrained model/tokenizer into models/classification/<model_name>/
    For full fine-tuning implement datasets tokenization and Trainer.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library required for LLM flows. Install `pip install transformers`.")

    import pandas as pd
    df = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset)
    if "message" not in df.columns or "label" not in df.columns:
        raise ValueError("LLM trainer expects 'message' and 'label' columns.")

    model_id = (hyperparams or {}).get("pretrained_model", "distilbert-base-uncased")
    num_labels = len(df["label"].unique())

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

    timestamp = _timestamp()
    model_name = model_name or f"llm_{timestamp}"
    model_dir = os.path.join(MODELS_DIR, "classification", model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    return {
        "task_type": "classification",
        "model_type": "llm",
        "model_name": model_name,
        "model_path": model_dir,
        "note": "LLM artifacts saved (scaffold). Implement full fine-tuning for real training.",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
    }


def build_chatbot_index(dataset_name: str = "faq", force: bool = False) -> Dict[str, Any]:
    """
    Build chatbot FAQ index (questions, answers, embeddings) saved as pickle.
    """
    model_dir = os.path.join(MODELS_DIR, "chatbot")
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, f"{dataset_name}_index.pkl")
    if os.path.exists(out_path) and not force:
        return {"index_path": out_path, "note": "exists"}

    from core.dataset_handler import load_chatbot_dataset
    items = load_chatbot_dataset(name=dataset_name)

    questions, answers = [], []
    for it in items:
        q = it.get("question") or it.get("q") or it.get("prompt") or ""
        a = it.get("answer") or it.get("response") or it.get("a") or ""
        if q and a:
            questions.append(str(q))
            answers.append(str(a))

    if SENTEVAL_AVAILABLE:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(questions, convert_to_tensor=False, show_progress_bar=False)
        embeddings = np.asarray(embeddings)
    else:
        # fallback: TF-IDF matrix
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=20000)
        embeddings = vec.fit_transform(questions).toarray()
        with open(os.path.join(model_dir, f"{dataset_name}_tfidf.pkl"), "wb") as f:
            pickle.dump(vec, f)

    with open(out_path, "wb") as f:
        pickle.dump((questions, answers, embeddings), f)

    return {"index_path": out_path, "n_q": len(questions)}


def build_knowledge_embeddings(force: bool = False) -> Dict[str, Any]:
    """
    Index knowledge docs (pdf/txt/md) into embeddings saved as pickle.
    """
    model_dir = os.path.join(MODELS_DIR, "knowledge")
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "embeddings.pkl")
    if os.path.exists(out_path) and not force:
        return {"embeddings_path": out_path, "note": "exists"}

    from core.dataset_handler import load_knowledge_dataset
    filepaths = load_knowledge_dataset()
    docs = []
    for fp in filepaths:
        try:
            if fp.lower().endswith(".pdf"):
                import PyPDF2
                with open(fp, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    txt = []
                    for p in reader.pages:
                        try:
                            txt.append(p.extract_text() or "")
                        except Exception:
                            txt.append("")
                    full = "\n".join(txt).strip()
                    if full:
                        docs.append(full)
                        continue
            # txt/md
            with open(fp, "r", encoding="utf-8") as f:
                docs.append(f.read())
        except Exception:
            docs.append(f"[File: {os.path.basename(fp)}]")

    if not docs:
        with open(out_path, "wb") as f:
            pickle.dump(([], np.array([])), f)
        return {"embeddings_path": out_path, "n_docs": 0}

    if SENTEVAL_AVAILABLE:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(docs, convert_to_tensor=False, show_progress_bar=False)
        embeddings = np.asarray(embeddings)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=20000)
        embeddings = vec.fit_transform(docs).toarray()
        with open(os.path.join(model_dir, "knowledge_tfidf.pkl"), "wb") as f:
            pickle.dump(vec, f)

    with open(out_path, "wb") as f:
        pickle.dump((docs, embeddings), f)

    return {"embeddings_path": out_path, "n_docs": len(docs)}


def ensure_trained_models():
    """
    Check default artifact presence; create defaults if missing.
    """
    # classification
    class_latest = _get_latest_model_file("classification", ext_filters=(".joblib", ".pkl"))
    if not class_latest:
        try:
            from core.dataset_handler import load_dataset
            print("⚙️ Training default classification (sms_spam)...")
            df = load_dataset("classification", dataset_name="sms_spam")
            train_classification(df, model_name="sms_spam_default", model_type="auto")
            print("✅ Default classification trained.")
        except Exception as e:
            print("⚠️ Default classification training failed:", e)

    # chatbot
    chat_latest = _get_latest_model_file("chatbot", ext_filters=(".pkl",))
    if not chat_latest:
        try:
            print("⚙️ Building chatbot index (faq)...")
            build_chatbot_index("faq")
            print("✅ Chatbot index built.")
        except Exception as e:
            print("⚠️ Chatbot build failed:", e)

    # knowledge
    know_latest = _get_latest_model_file("knowledge", ext_filters=(".pkl",))
    if not know_latest:
        try:
            print("⚙️ Building knowledge embeddings...")
            build_knowledge_embeddings()
            print("✅ Knowledge embeddings built.")
        except Exception as e:
            print("⚠️ Knowledge build failed:", e)


def load_trained_model(model_path: str):
    """
    Load model artifact. For joblib returns pipeline, for pickle returns tuple, for dir returns path.
    """
    if os.path.isdir(model_path):
        return model_path
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Model not found: {model_path}")
