# core/inference_engine.py
"""
Inference helpers for:
 - predict_classification(text, model_path=None)
 - chatbot_response(query, index_path=None)
 - knowledge_lookup(query, embeddings_path=None)

Behavior:
- If explicit path not provided, uses latest artifacts under models/<task>/
- Uses sentence-transformers for semantic search if available; otherwise uses TF-IDF fallback.
"""

import os
import pickle
import joblib
import numpy as np

# optional libs
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTEVAL_AVAILABLE = True
except Exception:
    SENTEVAL_AVAILABLE = False

# sklearn fallback for cosine
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_COSINE = True
except Exception:
    SKLEARN_COSINE = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _latest_in_dir(subdir: str, exts=(".joblib", ".pkl")):
    d = os.path.join(MODELS_DIR, subdir)
    if not os.path.exists(d):
        return None
    candidates = []
    for fname in os.listdir(d):
        if fname.endswith(exts):
            candidates.append(os.path.join(d, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# -------------------------
# Classification prediction
# -------------------------
def predict_classification(text: str, model_path: str = None):
    """
    Predict label for text. If model_path None, uses latest joblib in models/classification/.
    Returns predicted label (string).
    """
    if model_path is None:
        model_path = _latest_in_dir("classification", exts=(".joblib", ".pkl"))

    if model_path is None:
        raise FileNotFoundError("No classification model found. Train one first.")

    # if it's a joblib sklearn pipeline
    if model_path.endswith(".joblib"):
        pipe = joblib.load(model_path)
        return pipe.predict([text])[0]
    else:
        # legacy pickle (vectorizer, clf)
        with open(model_path, "rb") as f:
            vec, clf = pickle.load(f)
        X = vec.transform([text])
        return clf.predict(X)[0]


# -------------------------
# Chatbot response (FAQ)
# -------------------------
def chatbot_response(query: str, index_path: str = None) -> str:
    """
    Return best-matching answer from FAQ index.
    index_path defaults to latest index in models/chatbot/
    """
    if index_path is None:
        # prefer *_index.pkl
        index_path = _latest_in_dir("chatbot", exts=(".pkl",))

    if index_path is None:
        raise FileNotFoundError("No chatbot index found. Build one first with build_chatbot_index().")

    with open(index_path, "rb") as f:
        questions, answers, embeddings = pickle.load(f)

    # semantic search if sentence-transformers available and embeddings look dense
    if SENTEVAL_AVAILABLE:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = embedder.encode([query], convert_to_tensor=False)
        # compute cosine sim
        sims = np.dot(embeddings, np.array(query_emb).reshape(-1)) if embeddings.ndim == 2 and query_emb is not None else None
        # safer compute via util if available
        try:
            scores = st_util.cos_sim(query_emb, embeddings)[0]  # shape: (n,)
            best_idx = int(np.argmax(np.array(scores)))
            return answers[best_idx]
        except Exception:
            pass

    # fallback: sklearn cosine similarity or numpy
    if embeddings is not None:
        try:
            if embeddings.ndim == 2:
                if SKLEARN_COSINE:
                    qvec = np.array(query_emb) if 'query_emb' in locals() else None
                    if qvec is None:
                        # compute using TF-IDF fallback vectorizer if present
                        # attempt to load tfidf vectorizer file
                        vecpath = os.path.join(os.path.dirname(index_path), os.path.basename(index_path).replace("_index.pkl", "_tfidf.pkl"))
                        if os.path.exists(vecpath):
                            with open(vecpath, "rb") as vf:
                                vec = pickle.load(vf)
                            qvec = vec.transform([query]).toarray()[0]
                        else:
                            # cannot compute
                            return answers[0]
                    sims = cosine_similarity([qvec], embeddings)[0]
                else:
                    # simple dot-product fallback (may be unnormalized)
                    qvec = np.array(query_emb) if 'query_emb' in locals() else None
                    if qvec is None:
                        return answers[0]
                    sims = np.dot(embeddings, qvec)
                best_idx = int(np.argmax(sims))
                return answers[best_idx]
        except Exception:
            pass

    # as final fallback return first answer
    return answers[0] if answers else ""


# -------------------------
# Knowledge lookup
# -------------------------
def knowledge_lookup(query: str, embeddings_path: str = None) -> str:
    """
    Return best matching document snippet for query. If embeddings_path is None, uses latest in models/knowledge.
    """
    if embeddings_path is None:
        embeddings_path = _latest_in_dir("knowledge", exts=(".pkl",))
    if embeddings_path is None:
        raise FileNotFoundError("No knowledge embeddings found. Build them with build_knowledge_embeddings().")

    with open(embeddings_path, "rb") as f:
        docs, embeddings = pickle.load(f)

    if not docs:
        return "No knowledge documents available."

    # semantic search
    if SENTEVAL_AVAILABLE:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = embedder.encode([query], convert_to_tensor=False)
        try:
            scores = st_util.cos_sim(q_emb, embeddings)[0]
            best_idx = int(np.argmax(np.array(scores)))
            snippet = docs[best_idx]
            return snippet[:1000]  # return up to 1000 chars
        except Exception:
            pass

    # sklearn fallback
    if embeddings is not None:
        try:
            if embeddings.ndim == 2:
                if SKLEARN_COSINE:
                    qvec = None
                    # try to load tfidf used to produce embeddings (knowledge_tfidf.pkl)
                    vecpath = os.path.join(os.path.dirname(embeddings_path), "knowledge_tfidf.pkl")
                    if os.path.exists(vecpath):
                        with open(vecpath, "rb") as vf:
                            vec = pickle.load(vf)
                        qvec = vec.transform([query]).toarray()[0]
                        sims = cosine_similarity([qvec], embeddings)[0]
                        best_idx = int(np.argmax(sims))
                        return docs[best_idx][:1000]
                else:
                    return docs[0][:1000]
        except Exception:
            pass

    # fallback
    return docs[0][:1000]
