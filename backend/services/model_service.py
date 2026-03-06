import os
import joblib
import pickle
from functools import lru_cache
from typing import Any, Tuple, Dict, Optional
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE, "models")

def _find_model_path(model_id: str) -> str:
    if os.path.isabs(model_id) and os.path.exists(model_id):
        return model_id
    for sub in ("default", "custom"):
        folder = os.path.join(MODELS_DIR, sub)
        if not os.path.exists(folder):
            continue
        for ext in (".pkl", ".joblib", ".sav"):
            p = os.path.join(folder, model_id + ext)
            if os.path.exists(p):
                return p
        anyp = os.path.join(folder, model_id)
        if os.path.exists(anyp):
            return anyp
    for root, _, files in os.walk(MODELS_DIR):
        for f in files:
            if model_id in f:
                return os.path.join(root, f)
    raise FileNotFoundError(f"Model '{model_id}' not found in models/default or models/custom")

@lru_cache(maxsize=16)
def _load_model_cached(path: str):
    try:
        model = joblib.load(path)
        return model
    except Exception:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

def load_model_by_id(model_id: str):
    p = _find_model_path(model_id)
    return _load_model_cached(os.path.abspath(p))

def list_available_models() -> Dict[str, list]:
    out = {"default": [], "custom": []}
    for sub in ("default", "custom"):
        folder = os.path.join(MODELS_DIR, sub)
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        out[sub] = [os.path.splitext(f)[0] for f in files]
    return out

class ModelService:
    def __init__(self):
        self.models_dir = MODELS_DIR

    def predict(self, model_id: str, input_obj: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, float]], Dict[str, Any]]:
        model = load_model_by_id(model_id)
        X = None
        meta = {"input_shape": None, "model_type": type(model).__name__}

        if isinstance(input_obj, dict) and "text" in input_obj:
            X = pd.Series([input_obj["text"]])
        elif isinstance(input_obj, dict) and "features" in input_obj:
            X = np.array([input_obj["features"]])
        elif isinstance(input_obj, dict):
            X = pd.DataFrame([input_obj])
        else:
            X = pd.DataFrame([input_obj])

        meta["input_shape"] = getattr(X, "shape", None)

        try:
            if hasattr(model, "predict_proba"):
                preds = model.predict(X)
                probs = model.predict_proba(X)
                prob_dict = None
                if probs is not None and probs.shape[0] >= 1:
                    labels = getattr(model, "classes_", None)
                    if labels is not None and len(labels) == probs.shape[1]:
                        prob_dict = {str(labels[i]): float(probs[0, i]) for i in range(probs.shape[1])}
                    else:
                        prob_dict = {str(i): float(probs[0, i]) for i in range(probs.shape[1])}
                return (preds[0] if isinstance(preds, (list, tuple, np.ndarray)) else preds, prob_dict, meta)
            else:
                preds = model.predict(X)
                return (preds[0] if isinstance(preds, (list, tuple, np.ndarray)) else preds, None, meta)
        except Exception as e:
            if hasattr(model, "transform") and hasattr(model, "predict"):
                try:
                    Xt = model.transform(X)
                    preds = model.predict(Xt)
                    return (preds[0] if isinstance(preds, (list, tuple, np.ndarray)) else preds, None, meta)
                except Exception:
                    pass
            raise RuntimeError(f"Prediction failed: {str(e)}")
