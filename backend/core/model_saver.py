"""
core/model_saver.py

Phase 7 — Model packaging and export for ModelForge AI.

Primary function:
    export_path = save_model(model_info, build_runtime=False, runtime_size_threshold=200*1024*1024)

Expect model_info to include at least:
    - "model_path": path to the model artifact (joblib .pkl or similar)
    - "meta_path": path to the metadata JSON (optional but recommended)
    - "model_name": string
    - "task_type": one of "classification", "chatbot", "knowledge"

Package structure (zip):
    <model_name>_<timestamp>/
        artifact/                    # original model files (model_path, meta_path)
        runtime/ (optional)          # small inference helper + minimal assets
        manifest.json                # checksums + metadata
        README.txt

Notes:
- Does not upload to remote stores; hooks provided for register/upload integration.
- Avoids loading pickles unless building runtime (controlled by build_runtime).
"""
import os
import shutil
import json
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Use joblib only if runtime build requires it
try:
    import joblib
except Exception:
    joblib = None

EXPORTS_DIR = os.path.join("exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _sha256_of_file(path: str, chunk_size: int = 2 ** 20) -> str:
    """Compute SHA256 hex digest for a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: str, dst_dir: str) -> str:
    """Copy src into dst_dir, return final path."""
    _ensure_dir(dst_dir)
    fname = os.path.basename(src)
    dst = os.path.join(dst_dir, fname)
    shutil.copy2(src, dst)
    return dst


def _read_json_if_exists(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ----------------------------
# Runtime helper templates
# ----------------------------
_RUNTIME_LOADER_TEMPLATE = """\
# runtime_loader.py
# Minimal inference helper for {task_type} model: {model_name}
# This is a **lightweight** template. Adjust imports if your environment differs.

import joblib
import numpy as np
from scipy import sparse

MODEL_PATH = "{rel_model_file}"

def load_model():
    obj = joblib.load(MODEL_PATH)
    return obj

{task_infer_fn}
"""

_CLASSIFICATION_INFER_FN = """\
def predict_texts(texts):
    \"\"\"Return predicted labels for a list of texts.\"\"\"
    model_obj = load_model()
    vectorizer = model_obj.get('vectorizer') or model_obj.get('vectorizer_type')
    clf = model_obj.get('classifier') or model_obj.get('model')
    if vectorizer is None or clf is None:
        raise RuntimeError('Expected vectorizer and classifier stored in model object.')

    # If the vectorizer is a HashingVectorizer, it should also implement transform(...)
    X = vectorizer.transform(texts)
    preds = clf.predict(X)
    return preds.tolist()
"""

_CHATBOT_INFER_FN = """\
def get_answer(question, top_k=1):
    \"\"\"Return top_k answer(s) for a question (cosine similarity over vectorized FAQ).\"\"\"
    model_obj = load_model()
    vect = model_obj.get('vectorizer')
    if vect is None and model_obj.get('vectorizer_type') == 'hashing':
        vect = model_obj.get('vectorizer')  # hashing vectorizer may be a lightweight stateless object

    questions = model_obj.get('questions', [])
    answers = model_obj.get('answers', [])
    matrix = model_obj.get('matrix', None)

    # If matrix is a path (large-case), user should load it via scipy.sparse.load_npz
    if isinstance(matrix, str):
        from scipy import sparse
        matrix = sparse.load_npz(matrix)

    q_vec = vect.transform([question])
    sims = cosine_similarity(q_vec, matrix).flatten()
    idx = sims.argsort()[::-1][:top_k]
    return [answers[i] for i in idx]
"""

_KNOWLEDGE_INFER_FN = """\
def search_documents(query, top_k=5):
    \"\"\"Return top_k document indices and similarity scores for a query.\"\"\"
    model_obj = load_model()
    # try dense embeddings
    if 'embeddings' in model_obj and model_obj['embeddings'] is not None:
        import numpy as np
        encoder_name = model_obj.get('encoder_name')
        # If encoder not present in package, user should instantiate SentenceTransformer(encoder_name)
        emb = model_obj['embeddings']
        # We assume embeddings is an np.ndarray
        # A real implementation should re-encode the query using the same encoder.
        raise RuntimeError('This runtime bundle contains precomputed embeddings. To search, re-encode query using the same encoder and compute similarities.')
    # fallback to TF-IDF matrix
    vect = model_obj.get('vectorizer')
    doc_matrix = model_obj.get('doc_matrix')
    if isinstance(doc_matrix, str):
        from scipy import sparse
        doc_matrix = sparse.load_npz(doc_matrix)
    q_vec = vect.transform([query])
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_vec, doc_matrix).flatten()
    idx = sims.argsort()[::-1][:top_k]
    return [{"index": int(i), "score": float(sims[i])} for i in idx]
"""


# ----------------------------
# Core packaging function
# ----------------------------
def package_model(
    model_info: Dict,
    export_root: str = EXPORTS_DIR,
    build_runtime: bool = False,
    runtime_size_threshold: int = 200 * 1024 * 1024,
) -> str:
    """
    Package model artifacts into a zip file and return the zip path.

    Parameters
    ----------
    model_info : dict
        The dict returned by your trainer, with keys: model_path, meta_path, model_name, task_type
    build_runtime : bool
        If True, attempts to build a tiny runtime bundle (imports joblib and creates runtime_loader.py).
        Building runtime loads the model (joblib) which may be heavy; runtime_size_threshold controls auto-skip.
    runtime_size_threshold : int
        If the model artifact is larger than this threshold, runtime will not be built automatically
        unless build_runtime is explicitly True.
    """
    # Validate model_info
    model_path = model_info.get("model_path")
    meta_path = model_info.get("meta_path")
    model_name = model_info.get("model_name") or f"model_{int(time.time())}"
    task_type = model_info.get("task_type") or "unknown"

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    # prepare export directories
    export_task_dir = os.path.join(export_root, task_type)
    _ensure_dir(export_task_dir)

    stamp = _timestamp()
    bundle_name = f"{model_name}_{stamp}"
    bundle_dir = os.path.join(export_task_dir, bundle_name)
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    artifact_dir = os.path.join(bundle_dir, "artifact")
    os.makedirs(artifact_dir, exist_ok=True)

    # Copy model artifact & metadata (do not unpickle)
    copied_model = _copy_file(model_path, artifact_dir)
    copied_meta = None
    if meta_path and os.path.exists(meta_path):
        copied_meta = _copy_file(meta_path, artifact_dir)

    # Build manifest dictionary
    manifest = {
        "bundle_name": bundle_name,
        "model_name": model_name,
        "task_type": task_type,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {},
    }

    # compute checksums for artifact files
    for p in sorted(Path(artifact_dir).glob("*")):
        pth = str(p)
        manifest["files"][os.path.basename(pth)] = {
            "path": os.path.relpath(pth, bundle_dir),
            "size": os.path.getsize(pth),
            "sha256": _sha256_of_file(pth),
        }

    # Optionally create a runtime bundle (small inference helper + essential loaded items)
    runtime_dir = os.path.join(bundle_dir, "runtime")
    runtime_built = False
    if build_runtime:
        # check model size to avoid loading huge pickles unintentionally
        model_size = os.path.getsize(copied_model)
        if model_size > runtime_size_threshold and not build_runtime:
            # not building runtime automatically
            print(f"⚠️  Model artifact is large ({model_size} bytes). Skipping runtime build.")
        else:
            # Attempt to build runtime: joblib required
            if joblib is None:
                print("⚠️  joblib not available; cannot build runtime bundle.")
            else:
                try:
                    # load model object (may be heavy)
                    print("ℹ️  Loading model object to create runtime bundle (this may take memory)...")
                    model_obj = joblib.load(copied_model)
                    os.makedirs(runtime_dir, exist_ok=True)

                    # Save minimal parts for runtime:
                    # - For classification: vectorizer + classifier into runtime/artifact_model.pkl
                    # - For chatbot: vectorizer + questions/answers + (matrix path) else stored as model
                    # - For knowledge: store encoder_name + embeddings.npy (if present)
                    runtime_model_file = os.path.join(runtime_dir, os.path.basename(copied_model))
                    joblib.dump(model_obj, runtime_model_file, compress=1)

                    # Write runtime_loader.py using templates
                    rel_model_file = os.path.join("artifact", os.path.basename(copied_model))
                    if task_type == "classification":
                        task_fn = _CLASSIFICATION_INFER_FN
                    elif task_type == "chatbot":
                        task_fn = _CHATBOT_INFER_FN
                    elif task_type == "knowledge":
                        task_fn = _KNOWLEDGE_INFER_FN
                    else:
                        task_fn = "def _noop():\n    pass\n"

                    loader_code = _RUNTIME_LOADER_TEMPLATE.format(
                        task_type=task_type, model_name=model_name, rel_model_file=os.path.join("artifact", os.path.basename(copied_model)), task_infer_fn=task_fn
                    )
                    loader_path = os.path.join(runtime_dir, "runtime_loader.py")
                    with open(loader_path, "w", encoding="utf-8") as f:
                        f.write(loader_code)

                    # If model_obj contains large arrays as numpy arrays, optionally write them separately and reference
                    # (We avoid rewriting everything; runtime includes the joblib artifact we saved).
                    runtime_built = True
                    manifest["runtime"] = {"built": True, "runtime_files": [os.path.relpath(runtime_model_file, bundle_dir), os.path.relpath(loader_path, bundle_dir)]}
                except Exception as ex:
                    print(f"⚠️  Failed to build runtime bundle: {ex}")

    # Add README and manifest files
    readme_text = f"""ModelForge AI Export
====================
Model name: {model_name}
Task type: {task_type}
Bundle: {bundle_name}
Created: {manifest['created_at']}

Included files:
{json.dumps(manifest['files'], indent=2)}

Usage:
- Unzip the package.
- Inspect 'artifact/' for the original model pickle and metadata.
- If 'runtime/' exists, you can run runtime/runtime_loader.py after installing dependencies (joblib, scipy, sklearn).
"""
    with open(os.path.join(bundle_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme_text)

    # Save manifest JSON
    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Create the zip file
    zip_name = os.path.join(export_task_dir, f"{bundle_name}.zip")
    if os.path.exists(zip_name):
        os.remove(zip_name)

    # Make zip from bundle_dir contents (store without full absolute paths)
    shutil.make_archive(base_name=os.path.splitext(zip_name)[0], format="zip", root_dir=export_task_dir, base_dir=bundle_name)

    # return zip path
    print(f"✅ Packaged model into: {zip_name}")
    return zip_name


# ----------------------------
# Public wrapper expected by main pipeline
# ----------------------------
def save_model(model_info: Dict, build_runtime: bool = False, runtime_size_threshold: int = 200 * 1024 * 1024) -> str:
    """
    Save/package the model and return an exportable file path (zip).

    Parameters:
        model_info: dict returned by trainer (must have model_path and model_name)
        build_runtime: whether to attempt to create a runtime bundle (may load model)
        runtime_size_threshold: skip building runtime automatically if model larger than this
    """
    zip_path = package_model(model_info, build_runtime=build_runtime, runtime_size_threshold=runtime_size_threshold)
    # Optionally: register artifact in model registry here if you have one:
    try:
        from core.model_registry import register_model_artifact  # optional function
        try:
            register_model_artifact(model_info, zip_path)
        except Exception:
            # ignore registry failure gracefully
            pass
    except Exception:
        pass

    return zip_path


# ----------------------------
# CLI usage
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Package a trained model for export.")
    parser.add_argument("--model-path", help="Path to model artifact (.pkl) (required)", required=True)
    parser.add_argument("--meta-path", help="Path to meta JSON (optional)", required=False)
    parser.add_argument("--task-type", help="classification/chatbot/knowledge", required=True)
    parser.add_argument("--model-name", help="Model name (required)", required=True)
    parser.add_argument("--build-runtime", help="Attempt to build runtime bundle", action="store_true")
    args = parser.parse_args()

    info = {
        "model_path": args.model_path,
        "meta_path": args.meta_path,
        "model_name": args.model_name,
        "task_type": args.task_type,
    }
    out = save_model(info, build_runtime=args.build_runtime)
    print(json.dumps({"export": out}, indent=2))
