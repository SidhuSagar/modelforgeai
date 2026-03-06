# core/model_manager.py
"""
Model registry & packaging helpers.

Functions exposed:
 - list_models()
 - register_model_metadata(metadata)
 - find_models_on_disk(task_type=None)
 - load_model_metadata(model_name)
 - package_model_for_download(metadata_or_model_path, model_name=None, include_metadata=True) -> zip_path
 - find_package_for_model(model_name) -> zip_path or None
 - download_and_extract_package(package_path, dest_dir='.', extract=True) -> extracted_path
"""

import os
import json
import zipfile
import shutil
import time
from datetime import datetime
from typing import List, Dict, Optional, Union

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")
REGISTRY_PATH = os.path.join(OUTPUTS_ROOT, "registry.json")
PACKAGES_DIR = os.path.join(OUTPUTS_ROOT, "packages")

os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(OUTPUTS_ROOT, exist_ok=True)
os.makedirs(PACKAGES_DIR, exist_ok=True)


# -------------------------
# Registry helpers
# -------------------------
def _read_registry() -> List[Dict]:
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_registry(registry: List[Dict]):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def list_models() -> List[Dict]:
    """
    Return list of registered model metadata (reads outputs/registry.json).
    """
    return _read_registry()


def register_model_metadata(metadata: Dict):
    """
    Add metadata to registry (no duplicates by model_name).
    """
    registry = _read_registry()
    if not isinstance(registry, list):
        registry = []
    model_name = metadata.get("model_name")
    if model_name is None:
        # try to assign auto name
        model_name = f"model_{int(time.time())}"
        metadata["model_name"] = model_name

    exists = any(item.get("model_name") == model_name for item in registry)
    if not exists:
        registry.append(metadata)
        _write_registry(registry)


def load_model_metadata(model_name: str) -> Optional[Dict]:
    """
    Return metadata for the given model_name from registry, or None.
    """
    registry = _read_registry()
    for item in registry:
        if item.get("model_name") == model_name:
            return item
    return None


# -------------------------
# Disk helpers
# -------------------------
def find_models_on_disk(task_type: str = None) -> List[str]:
    """
    Return a list of model artifact paths found under models/ (optionally filtered by task_type subfolder).
    """
    paths = []
    root = MODELS_ROOT
    if task_type:
        root = os.path.join(MODELS_ROOT, task_type)
    if not os.path.exists(root):
        return paths
    for root_dir, _, files in os.walk(root):
        for f in files:
            if f.endswith((".joblib", ".pkl")) or os.path.isdir(os.path.join(root_dir, f)):
                paths.append(os.path.join(root_dir, f))
    return paths


# -------------------------
# Packaging for download
# -------------------------
def _generate_readme(metadata: Dict, model_path: str) -> str:
    """Generate README.md content for exported model package."""
    task_type = metadata.get("task", "classification")
    model_name = metadata.get("model_name", os.path.basename(model_path))
    
    readme = f"""# ModelForge AI - Model Package

## Model Information

- **Model Name**: {model_name}
- **Task Type**: {task_type}
- **Created**: {metadata.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
- **Model File**: {os.path.basename(model_path)}

## How to Run This Model Locally

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Required packages**:
   ```bash
   pip install scikit-learn pandas numpy joblib
   ```
   
   For knowledge base models, also install:
   ```bash
   pip install sentence-transformers
   ```

### Installation Steps

1. **Extract the ZIP file** to a directory of your choice
   ```bash
   unzip {model_name}_*.zip
   cd {model_name}_*
   ```

2. **Install dependencies** (if not already installed)
   ```bash
   pip install scikit-learn pandas numpy joblib
   ```

3. **Use the model** in your Python code:

#### For Classification Models

```python
from core.predictor import Predictor
import os

# Initialize predictor
predictor = Predictor()

# Load the model (adjust path if needed)
model_file = "{os.path.basename(model_path)}"
predictor.load(model_file)

# Make a prediction
result = predictor.predict_text("Your text here")
print(result)
# Output: {{"task": "classification", "label": "...", "probabilities": {{...}}}}
```

#### For Chatbot Models

```python
from core.predictor import Predictor

predictor = Predictor()
predictor.load("{os.path.basename(model_path)}")

# Query the chatbot
result = predictor.predict_text("What is AI?", top_k=3)
print(result)
# Output: {{"task": "chatbot", "results": [{{"question": "...", "answer": "...", "score": 0.95}}]}}
```

#### For Knowledge Base Models

```python
from core.predictor import Predictor

predictor = Predictor()
predictor.load("{os.path.basename(model_path)}")

# Query the knowledge base
result = predictor.predict_text("Your question here", top_k=3)
print(result)
# Output: {{"task": "knowledge", "results": [{{"text": "...", "score": 0.95}}]}}
```

### Alternative: Using scikit-learn Directly (Classification Only)

If you have a classification model, you can also load it directly:

```python
import joblib

# Load the model
model = joblib.load("{os.path.basename(model_path)}")

# Make prediction
prediction = model.predict(["Your text here"])
print(prediction)
```

### Notes

- The model file (`{os.path.basename(model_path)}`) must be in the same directory as your script, or provide the full path
- For best results with knowledge base models, ensure `sentence-transformers` is installed
- Model metadata is available in `metadata.json` if included in the package

### Troubleshooting

1. **Import Error**: Make sure you have the ModelForge AI `core` module in your Python path, or copy the `core` folder from the ModelForge AI project
2. **File Not Found**: Ensure the model file is in the correct location
3. **Dependencies Missing**: Install all required packages listed above

### Support

For more information, visit the ModelForge AI documentation or check the `abtproject.md` file in the main project directory.

---
Generated by ModelForge AI v2.2
"""
    return readme


def package_model_for_download(metadata_or_model_path: Union[Dict, str], model_name: Optional[str] = None, include_metadata: bool = True) -> str:
    """
    Package a trained model (folder or single file) into a zip in outputs/packages/.

    Args:
        metadata_or_model_path: either:
            - metadata dict containing at least 'model_path' and 'model_name'
            - or direct path to model artifact (file or folder)
        model_name: optional override for naming the zip. If not provided, metadata/model's basename used.
        include_metadata: if True and metadata dict provided or found in registry, write metadata.json into zip.

    Returns:
        zip_path: absolute path to created .zip file
    """
    # resolve model_path and metadata
    metadata = None
    model_path = None
    if isinstance(metadata_or_model_path, dict):
        metadata = metadata_or_model_path
        model_path = metadata.get("model_path")
        model_name = model_name or metadata.get("model_name")
    else:
        model_path = metadata_or_model_path
        model_name = model_name or os.path.splitext(os.path.basename(model_path))[0]

        # try to find registry metadata if exists
        found = load_model_metadata(model_name)
        if found:
            metadata = found

    if not model_path:
        raise ValueError("model_path could not be resolved for packaging.")

    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_basename = f"{model_name}_{timestamp}"
    zip_path = os.path.join(PACKAGES_DIR, f"{zip_basename}.zip")

    # remove existing if any (rare)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # if model_path is a directory (HF model dir etc.) zip entire directory
    if os.path.isdir(model_path):
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', root_dir=model_path)
        # add metadata.json and README.md to zip if requested
        if include_metadata:
            meta_to_write = metadata or load_model_metadata(model_name) or {}
            with zipfile.ZipFile(zip_path, 'a', compression=zipfile.ZIP_DEFLATED) as zf:
                if metadata:
                    zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                readme_content = _generate_readme(meta_to_write, model_path)
                zf.writestr("README.md", readme_content)
        return zip_path

    # single-file case (e.g., .joblib or .pkl)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # write the model file
        zf.write(model_path, arcname=os.path.basename(model_path))
        # attempt to include sibling pkl or joblib variants if present
        folder = os.path.dirname(model_path)
        base = os.path.splitext(os.path.basename(model_path))[0]
        for ext in (".pkl", ".joblib"):
            sibling = os.path.join(folder, f"{base}{ext}")
            if os.path.exists(sibling) and os.path.abspath(sibling) != os.path.abspath(model_path):
                zf.write(sibling, arcname=os.path.basename(sibling))
        # include metadata.json if available
        if include_metadata:
            meta_to_write = metadata or load_model_metadata(model_name) or {}
            zf.writestr("metadata.json", json.dumps(meta_to_write, indent=2))
        
        # Include README.md with usage instructions
        readme_content = _generate_readme(meta_to_write if include_metadata else (metadata or {}), model_path)
        zf.writestr("README.md", readme_content)

    return zip_path


def find_package_for_model(model_name: str) -> Optional[str]:
    """
    Search outputs/packages/ for the latest package matching model_name prefix.
    Returns package path or None.
    """
    if not model_name:
        return None
    candidates = []
    for fname in os.listdir(PACKAGES_DIR):
        if fname.lower().startswith(model_name.lower()):
            candidates.append(os.path.join(PACKAGES_DIR, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def download_and_extract_package(package_path: str, dest_dir: str = ".", extract: bool = True) -> str:
    """
    Copy a package (zip) to dest_dir and optionally extract it.
    Returns path to extracted folder (or copied zip path if extract=False).
    """
    if not os.path.exists(package_path):
        raise FileNotFoundError(f"Package not found: {package_path}")

    os.makedirs(dest_dir, exist_ok=True)
    dest_zip = os.path.join(os.path.abspath(dest_dir), os.path.basename(package_path))
    shutil.copyfile(package_path, dest_zip)

    if not extract:
        return dest_zip

    # extract into folder (same name as zip without extension)
    extract_folder = os.path.join(os.path.abspath(dest_dir), os.path.splitext(os.path.basename(package_path))[0])
    # ensure empty or create
    if os.path.exists(extract_folder):
        # do not overwrite by default - create new folder with timestamp suffix
        extract_folder = extract_folder + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(extract_folder, exist_ok=True)
    shutil.unpack_archive(dest_zip, extract_folder)
    return extract_folder


# -------------------------
# Convenience CLI helper
# -------------------------
def download_model(model_name: Optional[str] = None, package_path: Optional[str] = None, dest_dir: str = ".", extract: bool = True) -> str:
    """
    High-level helper:
      - If package_path provided: use it.
      - Else if model_name provided: find latest package for model_name.
      - Else raise.

    Copies package to dest_dir and extracts it (if extract=True). Returns path.
    """
    pkg = None
    if package_path:
        pkg = package_path
    elif model_name:
        pkg = find_package_for_model(model_name)
        if not pkg:
            # maybe model hasn't been packaged yet - try to package from registry
            meta = load_model_metadata(model_name)
            if meta and meta.get("model_path"):
                pkg = package_model_for_download(meta, model_name=model_name)
    else:
        raise ValueError("Provide either package_path or model_name")

    if not pkg or not os.path.exists(pkg):
        raise FileNotFoundError("Package not found or could not be created.")

    result = download_and_extract_package(pkg, dest_dir=dest_dir, extract=extract)
    return result
 