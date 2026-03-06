# api_server.py  — Model Forge AI Backend (v2.2)
import os
import uuid
import time
import json
import traceback
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Core Imports
# -------------------------
from core.dataset_handler import load_dataset
from core.model_trainer import (
    train_classification,
    build_chatbot_index,
    build_knowledge_embeddings,
)
from core.model_manager import register_model_metadata, package_model_for_download
from core.visualizer import visualize_training_summary
from core.predictor import Predictor

# -------------------------
# Setup FastAPI App
# -------------------------
app = FastAPI(
    title="🧠 Model Forge AI API",
    version="2.2",
    description="Backend API for training, tracking, and downloading AI models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://modelforgeai-k22i.vercel.app",
        "*",  # Allow all origins for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Constants & Globals
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIRS = [
    os.path.join(BASE_DIR, "outputs", "packages"),
    os.path.join(BASE_DIR, "models", "classification"),
    os.path.join(BASE_DIR, "models", "chatbot"),
    os.path.join(BASE_DIR, "models", "knowledge"),
]

JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Predictor cache: keyed by file path -> Predictor instance + metadata
_PREDICTOR_CACHE: Dict[str, Dict[str, Any]] = {}
# last loaded model path (for convenience / default)
_LAST_LOADED_MODEL: Optional[str] = None


# -------------------------
# Job helpers (unchanged)
# -------------------------
def _register_job(status="queued"):
    job_id = str(uuid.uuid4())
    JOB_REGISTRY[job_id] = {
        "status": status,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": [],
        "result": None,
    }
    return job_id


def _update_job(job_id: str, step: str, message: str, status="in-progress"):
    JOB_REGISTRY[job_id]["status"] = status
    JOB_REGISTRY[job_id]["steps"].append({"step": step, "message": message})
    print(f"[{job_id}] {step} → {message}")


def _finish_job(job_id: str, result: dict, status="completed"):
    JOB_REGISTRY[job_id]["status"] = status
    JOB_REGISTRY[job_id]["result"] = result


# -------------------------
# Model helper utilities
# -------------------------
def _find_model_files() -> List[str]:
    files = []
    for folder in MODEL_DIRS:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            full = os.path.join(folder, fname)
            if os.path.isfile(full):
                files.append(os.path.abspath(full))
    # sort by mtime desc
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _load_predictor_for_path(path: str) -> Dict[str, Any]:
    """Load predictor from cache or disk. Returns metadata including 'task'."""
    global _PREDICTOR_CACHE, _LAST_LOADED_MODEL
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if path in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[path]

    predictor = Predictor()
    meta = predictor.load(path)  # may raise
    entry = {"predictor": predictor, "meta": meta, "path": path, "loaded_at": time.time()}
    _PREDICTOR_CACHE[path] = entry
    _LAST_LOADED_MODEL = path
    return entry


# -------------------------
# Step 1️⃣ - Get Task Options
# -------------------------
@app.get("/tasks")
def get_task_types():
    return {
        "step": 1,
        "title": "Select your task type",
        "options": ["classification", "chatbot", "knowledge"],
    }


# -------------------------
# Step 2️⃣ - Get Model Options
# -------------------------
@app.get("/models")
def get_model_options():
    return {
        "step": 2,
        "title": "Choose your model type",
        "options": ["auto", "logisticregression", "randomforest", "svm", "llm"],
    }


# -------------------------
# Step 3️⃣ - Upload Dataset
# -------------------------
@app.post("/datasets/upload")
async def upload_dataset(task_type: str = Form(...), file: UploadFile = File(...)):
    folder = os.path.join(DATA_DIR, task_type)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # For knowledge base, just return file path (no cleaning needed)
    if task_type == "knowledge":
        return {
            "step": 3,
            "message": "✅ Knowledge base file uploaded",
            "file_path": file_path,
            "preview": {"filename": file.filename, "size": os.path.getsize(file_path)},
        }
    
    # Try cleaning dataset for classification/chatbot
    try:
        dataset = load_dataset(task_type, custom_path=file_path)
        preview = dataset.head(5).to_dict() if hasattr(dataset, "head") else None
        return {
            "step": 3,
            "message": "✅ Dataset uploaded and cleaned",
            "file_path": file_path,
            "preview": preview,
        }
    except Exception as e:
        return {
            "step": 3,
            "message": f"✅ File uploaded (warning: {str(e)})",
            "file_path": file_path,
            "preview": None,
        }


# -------------------------
# Step 4️⃣ - Preprocess settings
# -------------------------
@app.post("/preprocess")
def preprocess_settings(test_split: float = Form(0.2)):
    return {
        "step": 4,
        "message": f"✅ Test split ratio set to {test_split}",
        "test_split": test_split,
    }


# -------------------------
# Step 5️⃣ - Start Training Job (background)
# -------------------------
@app.post("/train/start")
def start_training(
    background_tasks: BackgroundTasks,
    task_type: str = Form(...),
    model_type: str = Form("auto"),
    dataset_path: str = Form(...),
    test_split: float = Form(0.2),
    epochs: int = Form(3),
):
    job_id = _register_job()
    _update_job(job_id, "Training Start", f"Task: {task_type}, Model: {model_type}")

    # Run background job
    background_tasks.add_task(
        _train_background_job,
        job_id,
        task_type,
        model_type,
        dataset_path,
        test_split,
        epochs,
    )

    return {
        "job_id": job_id,
        "status": "started",
        "status_url": f"/train/status/{job_id}",
        "message": "🚀 Training started in background",
    }


def _train_background_job(
    job_id: str,
    task_type: str,
    model_type: str,
    dataset_path: str,
    test_split: float,
    epochs: int,
):
    try:
        _update_job(job_id, "Dataset Loading", "Loading dataset...")
        dataset = load_dataset(task_type, custom_path=dataset_path)

        _update_job(job_id, "Model Training", f"Training {model_type.upper()} model...")
        if task_type == "classification":
            model_info = train_classification(
                dataset=dataset,
                model_type=model_type,
                test_size=test_split,
                hyperparams={"epochs": epochs},
            )
        elif task_type == "chatbot":
            model_info = build_chatbot_index()
        else:  # knowledge
            # For knowledge base, we need to rebuild embeddings with uploaded files
            model_info = build_knowledge_embeddings(force=True)

        # Visualization
        if task_type == "classification" and isinstance(model_info, dict) and "report" in model_info:
            _update_job(job_id, "Visualization", "Creating performance plots...")
            plot_path = visualize_training_summary(model_info)
            model_info["plot_path"] = os.path.abspath(plot_path)

        # Packaging
        _update_job(job_id, "Packaging", "Saving and packaging model...")
        register_model_metadata(model_info)
        zip_path = package_model_for_download(model_info)
        model_info["zip_path"] = os.path.abspath(zip_path)

        _finish_job(job_id, model_info, status="completed")
        print(f"✅ [JOB {job_id}] Training finished successfully.")

    except Exception as e:
        error = traceback.format_exc()
        JOB_REGISTRY[job_id]["status"] = "failed"
        JOB_REGISTRY[job_id]["error"] = str(e)
        JOB_REGISTRY[job_id]["traceback"] = error
        print(f"❌ [JOB {job_id}] Failed: {e}\n{error}")


# -------------------------
# Step 6️⃣ - Check Job Status
# -------------------------
@app.get("/train/status/{job_id}")
def check_training_status(job_id: str):
    job = JOB_REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# -------------------------
# Model listing & loading endpoints (NEW)
# -------------------------
@app.get("/models/list")
def list_models():
    files = _find_model_files()
    # Return relative paths to base dir for easier consumption in frontend
    display = [{"path": f, "name": os.path.basename(f), "mtime": os.path.getmtime(f)} for f in files]
    return {"count": len(display), "models": display}


@app.post("/models/load")
def load_model(model_path: str = Form(...)):
    """Load a model into server memory (cached Predictor). Returns metadata."""
    try:
        entry = _load_predictor_for_path(model_path)
        return {"ok": True, "message": "model loaded", "meta": entry["meta"], "path": entry["path"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/current")
def current_model():
    if _LAST_LOADED_MODEL is None:
        return {"loaded": False}
    meta = _PREDICTOR_CACHE.get(_LAST_LOADED_MODEL)
    if not meta:
        return {"loaded": False}
    return {"loaded": True, "path": _LAST_LOADED_MODEL, "meta": meta["meta"]}


# -------------------------
# Prediction endpoints (single and batch)
# -------------------------
@app.post("/predict")
async def predict_text(text: str = Form(...), model_path: Optional[str] = Form(None), top_k: int = Form(3)):
    """
    Predict single text.
    If model_path not supplied, server will attempt to use the last-loaded model.
    """
    global _LAST_LOADED_MODEL
    try:
        selected_path = model_path or _LAST_LOADED_MODEL
        if not selected_path:
            raise HTTPException(status_code=400, detail="No model_path provided and no model loaded. Call /models/load first.")

        entry = _load_predictor_for_path(selected_path)
        predictor = entry["predictor"]
        result = predictor.predict_text(text, top_k=top_k)
        return {"ok": True, "model": os.path.basename(selected_path), "result": result}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/predict/batch")
async def predict_batch(samples: List[str] = Form(...), model_path: Optional[str] = Form(None), top_k: int = Form(3)):
    """
    Accepts a list of strings (Form) or you can send as JSON body using fetch/axios (content-type: application/json)
    Example JSON body: {"samples": ["a","b","c"], "model_path": "..."}
    """
    try:
        # If sent as form list, FastAPI will convert; handle JSON body too:
        # NOTE: when using fetch with JSON set content-type application/json, FastAPI will not call here,
        # you should POST to /predict/batch with JSON body; see frontend example below.
        global _LAST_LOADED_MODEL
        selected_path = model_path or _LAST_LOADED_MODEL
        if not selected_path:
            raise HTTPException(status_code=400, detail="No model_path provided and no model loaded. Call /models/load first.")
        entry = _load_predictor_for_path(selected_path)
        predictor = entry["predictor"]
        results = [predictor.predict_text(s, top_k=top_k) for s in samples]
        return {"ok": True, "model": os.path.basename(selected_path), "results": results}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -------------------------
# Step 7️⃣ - Download Model File
# -------------------------
@app.get("/download/{filename}")
def download_model(filename: str):
    search_dirs = [
        os.path.join(BASE_DIR, "outputs", "packages"),
        os.path.join(BASE_DIR, "outputs", "plots"),  # Add plots directory
        os.path.join(BASE_DIR, "models", "classification"),
        os.path.join(BASE_DIR, "models", "chatbot"),
        os.path.join(BASE_DIR, "models", "knowledge"),
    ]
    for folder in search_dirs:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return FileResponse(path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


# -------------------------
# Health & Root
# -------------------------
@app.get("/")
def home():
    return {"message": "🧠 Model Forge AI API is running", "version": "2.2"}


@app.get("/health")
def health():
    return {"status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S")}


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
