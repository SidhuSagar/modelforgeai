# core/predictor.py
"""
Predictor helper used by main.py interactive testing phase.

Supported:
 - Classification models (joblib .joblib / .pkl pipelines or estimators)
 - Chatbot indexes (pickle files with 'questions','answers','vectorizer','vectors' or 'pairs')
 - Knowledge embeddings (pickle with [{'text':..., 'embedding':[...]}, ...] or plain docs)

Usage:
  from core.predictor import Predictor
  p = Predictor()
  p.load("path/to/model_or_index.pkl")
  p.predict_text("sample text")
"""

import os
import pickle
import joblib
import json
import math
import numpy as np
from typing import Any, Dict, List
from difflib import get_close_matches

try:
    # optional -- used for TF-IDF fallback similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


class Predictor:
    def __init__(self):
        self.task = None  # 'classification' | 'chatbot' | 'knowledge'
        self.model = None  # sklearn/pipeline or custom object
        self.vectorizer = None  # tfidf or similar (chatbot/knowledge fallback)
        self.vectors = None  # np.array of vectors (chatbot)
        self.qa_pairs = None  # fallback pairs list[(q,a), ...]
        self.embedded_docs = None  # list of dicts {'text':..., 'embedding':[...]}

    def _is_file(self, path: str) -> bool:
        return path and os.path.exists(path) and os.path.isfile(path)

    def load(self, model_path: str) -> Dict[str, Any]:
        """Load a model/index/embeddings. Returns metadata dict."""
        if not model_path:
            raise ValueError("No model_path provided.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # if path is a directory, try to find common filenames
        if os.path.isdir(model_path):
            # try common names
            candidates = [
                os.path.join(model_path, "model.joblib"),
                os.path.join(model_path, "model.pkl"),
                os.path.join(model_path, "index.pkl"),
                os.path.join(model_path, "faq_index.pkl"),
                os.path.join(model_path, "embeddings.pkl"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        meta = {"loaded_from": model_path}

        # 1) Try sklearn/joblib model (classification)
        try:
            if model_path.endswith(".joblib") or model_path.endswith(".pkl") or model_path.endswith(".model"):
                # attempt to load with joblib first (handles sklearn pipelines nicely)
                try:
                    obj = joblib.load(model_path)
                    # heuristics: if obj has predict method and classes_ or named_steps -> classification
                    if hasattr(obj, "predict"):
                        self.model = obj
                        self.task = "classification"
                        meta["task"] = "classification"
                        return meta
                except Exception:
                    # fall back to pickle
                    with open(model_path, "rb") as f:
                        obj = pickle.load(f)

                    # If obj looks like a dict-based index (chatbot/embeddings)
                    if isinstance(obj, dict):
                        # chatbot index detection
                        if "questions" in obj and "answers" in obj:
                            self.task = "chatbot"
                            self.qa_pairs = list(zip(obj.get("questions", []), obj.get("answers", [])))
                            if "vectorizer" in obj and "vectors" in obj:
                                self.vectorizer = obj["vectorizer"]
                                self.vectors = np.asarray(obj["vectors"])
                            meta["task"] = "chatbot"
                            return meta
                        # embeddings detection
                        if isinstance(obj.get("docs", None), list) or isinstance(obj.get("embeddings", None), list):
                            self.task = "knowledge"
                            self.embedded_docs = obj.get("docs") or obj.get("embeddings")
                            meta["task"] = "knowledge"
                            return meta
                        # else it could be a saved sklearn estimator by pickle
                        if hasattr(obj, "predict"):
                            self.model = obj
                            self.task = "classification"
                            meta["task"] = "classification"
                            return meta

                    # obj might be a list of dicts (embeddings)
                    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict) and "text" in obj[0]:
                        self.task = "knowledge"
                        self.embedded_docs = obj
                        meta["task"] = "knowledge"
                        return meta

            # else try joblib generic load
            try:
                obj = joblib.load(model_path)
                if hasattr(obj, "predict"):
                    self.model = obj
                    self.task = "classification"
                    meta["task"] = "classification"
                    return meta
            except Exception:
                pass

        except Exception:
            # continue to other loaders
            pass

        # 2) If file is a pickle index for chatbot or embeddings
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                # chatbot structure
                if "questions" in obj and "answers" in obj:
                    self.task = "chatbot"
                    self.qa_pairs = list(zip(obj.get("questions", []), obj.get("answers", [])))
                    if "vectorizer" in obj and "vectors" in obj:
                        self.vectorizer = obj["vectorizer"]
                        self.vectors = np.asarray(obj["vectors"])
                    meta["task"] = "chatbot"
                    return meta
                # embeddings structure
                if "embeddings" in obj and isinstance(obj["embeddings"], list):
                    self.task = "knowledge"
                    # unify format to [{'text':..., 'embedding':[...]}, ...]
                    emb = obj["embeddings"]
                    if isinstance(emb, list) and len(emb) and isinstance(emb[0], dict):
                        self.embedded_docs = emb
                    else:
                        # if embeddings and texts separate
                        texts = obj.get("texts") or obj.get("docs")
                        if texts and len(texts) == len(emb):
                            self.embedded_docs = [{"text": t, "embedding": e} for t, e in zip(texts, emb)]
                    meta["task"] = "knowledge"
                    return meta
                # fallback pair list
                if "pairs" in obj:
                    self.task = "chatbot"
                    self.qa_pairs = obj["pairs"]
                    meta["task"] = "chatbot"
                    return meta

            if isinstance(obj, list) and len(obj) > 0:
                if isinstance(obj[0], dict) and "text" in obj[0]:
                    self.task = "knowledge"
                    self.embedded_docs = obj
                    meta["task"] = "knowledge"
                    return meta

        except Exception:
            pass

        # 3) If none of the above, raise
        raise ValueError("Could not determine model/index type from file. "
                         "Supported: sklearn joblib/pickle classifier, chatbot index pickle, or embeddings pickle.")

    def predict_text(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """Return a prediction dict for the given text depending on loaded task."""
        if not text:
            return {"error": "empty input"}
        if self.task is None:
            raise RuntimeError("No model loaded. Call load(model_path) first.")

        if self.task == "classification":
            return self._predict_classification(text)
        elif self.task == "chatbot":
            return self._predict_chatbot(text, top_k=top_k)
        elif self.task == "knowledge":
            return self._predict_knowledge(text, top_k=top_k)
        else:
            return {"error": f"unknown task {self.task}"}

    def _predict_classification(self, text: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Classification model is not loaded.")
        try:
            # if pipeline supports predict_proba
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba([text])[0]
                pred = self.model.predict([text])[0]
                # build mapping of class->prob
                classes = getattr(self.model, "classes_", None)
                probs = {}
                if classes is not None and len(proba) == len(classes):
                    for c, p in zip(classes, proba):
                        probs[str(c)] = float(p)
                else:
                    probs = {"raw": proba.tolist()}
                return {"task": "classification", "label": str(pred), "probabilities": probs}
            else:
                pred = self.model.predict([text])[0]
                return {"task": "classification", "label": str(pred)}
        except Exception as e:
            return {"error": f"classification prediction failed: {e}"}

    def _predict_chatbot(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        # If vectorizer + vectors exist, do TF-IDF vector similarity
        if self.vectorizer is not None and self.vectors is not None:
            try:
                q_vec = self.vectorizer.transform([text])
                if hasattr(q_vec, "toarray"):
                    q_arr = q_vec.toarray()
                else:
                    q_arr = np.asarray(q_vec)
                sims = cosine_similarity(q_arr, self.vectors)[0] if cosine_similarity else \
                       [ _cosine(q_arr[0], v) for v in self.vectors ]
                idxs = np.argsort(sims)[::-1][:top_k]
                results = []
                for i in idxs:
                    results.append({"question": self.qa_pairs[i][0], "answer": self.qa_pairs[i][1], "score": float(sims[i])})
                return {"task": "chatbot", "results": results}
            except Exception as e:
                # fallback to fuzzy
                pass

        # fallback fuzzy matching using question strings
        if self.qa_pairs:
            questions = [q for q, _ in self.qa_pairs]
            # use difflib to find closest matches
            matches = get_close_matches(text, questions, n=top_k, cutoff=0.1)
            results = []
            for m in matches:
                try:
                    idx = questions.index(m)
                    results.append({"question": m, "answer": self.qa_pairs[idx][1], "score": None})
                except Exception:
                    continue
            return {"task": "chatbot", "results": results}
        return {"task": "chatbot", "results": [], "note": "no index content loaded"}

    def _predict_knowledge(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        if not self.embedded_docs:
            return {"task": "knowledge", "results": [], "note": "no docs loaded"}

        # If docs include precomputed embeddings:
        if all(isinstance(d.get("embedding", None), (list, tuple, np.ndarray)) for d in self.embedded_docs):
            # we need an embedding for the query. If a vectorizer/encoder was saved, we could call it,
            # but without that we attempt a TF-IDF fallback: compute TF-IDF vectors for all texts + query.
            try:
                texts = [d["text"] for d in self.embedded_docs]
                if TfidfVectorizer is None:
                    raise RuntimeError("sklearn not available for TF-IDF fallback.")
                tf = TfidfVectorizer().fit(texts + [text])
                doc_vecs = tf.transform(texts).toarray()
                q_vec = tf.transform([text]).toarray()[0]
                sims = [float(_cosine(q_vec, v)) for v in doc_vecs]
                idxs = np.argsort(sims)[::-1][:top_k]
                return {"task": "knowledge", "results": [{"text": texts[i], "score": float(sims[i])} for i in idxs]}
            except Exception as e:
                # fallback to simple substring matching
                pass

        # If no embeddings, do string-similarity via TF-IDF if available
        try:
            texts = [d.get("text", str(d)) for d in self.embedded_docs]
            if TfidfVectorizer is None:
                # basic substring match fallback
                matches = []
                for t in texts:
                    score = 1.0 if text.lower() in t.lower() else 0.0
                    matches.append(score)
                idxs = np.argsort(matches)[::-1][:top_k]
                return {"task": "knowledge", "results": [{"text": texts[i], "score": float(matches[i])} for i in idxs]}
            tf = TfidfVectorizer().fit(texts + [text])
            doc_vecs = tf.transform(texts)
            q_vec = tf.transform([text])
            sims = (cosine_similarity(q_vec, doc_vecs)[0]).tolist()
            idxs = np.argsort(sims)[::-1][:top_k]
            return {"task": "knowledge", "results": [{"text": texts[i], "score": float(sims[i])} for i in idxs]}
        except Exception as e:
            return {"task": "knowledge", "results": [], "error": str(e)}
