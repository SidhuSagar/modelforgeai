# phase5_utils/trainer_interface.py
"""
Unified training interface used by Phase 4/5 CLI.

train_and_register(task_type, dataset_name=None, model_name=None, custom=False,
                   model_type="auto", dataset_obj=None, train_config=None)

Supports:
 - classification (full)
 - chatbot (build index)
 - knowledge (build embeddings)
Packages artifacts and registers metadata via core.model_manager.
"""
import os
from typing import Optional, Dict, Any

from core.model_manager import register_model_metadata, package_model_for_download
from core.model_trainer import (
    train_classification,
    build_chatbot_index,
    build_knowledge_embeddings
)
from core.dataset_handler import load_dataset, load_custom_dataset

# optional external registry
try:
    from core.model_registry import register_model as register_model_registry
except Exception:
    register_model_registry = None


def train_and_register(task_type: str,
                       dataset_name: Optional[str] = None,
                       model_name: Optional[str] = None,
                       custom: bool = False,
                       *,
                       model_type: str = "auto",
                       dataset_obj=None,
                       train_config: Optional[Dict] = None) -> Dict[str, Any]:
    train_config = train_config or {}
    task_type = (task_type or "").strip().lower()
    metadata = {}

    if task_type == "classification":
        # resolve dataset
        dataset = None
        if custom:
            if dataset_obj is None and dataset_name:
                # treat dataset_name as path
                dataset = load_custom_dataset(dataset_name)
            elif dataset_obj is not None:
                dataset = dataset_obj
            else:
                raise ValueError("Custom dataset selected but no dataset_obj/path provided.")
        else:
            dataset = load_dataset("classification", dataset_name=dataset_name or "sms_spam")

        model_info = train_classification(
            dataset=dataset,
            model_name=model_name,
            model_type=model_type,
            test_size=train_config.get("test_size", 0.2),
            random_state=train_config.get("random_state", 42),
            hyperparams=train_config.get("hyperparams", None)
        )
        metadata = model_info

    elif task_type == "chatbot":
        dataset_to_use = dataset_name or "faq"
        res = build_chatbot_index(dataset_name=dataset_to_use, force=False)
        metadata = {
            "task_type": "chatbot",
            "model_type": "embedding_index",
            "model_name": f"{dataset_to_use}_faq_index",
            "model_path": res.get("index_path"),
            "n_q": res.get("n_q"),
            "trained_at": res.get("trained_at", None)
        }

    elif task_type == "knowledge":
        res = build_knowledge_embeddings(force=False)
        metadata = {
            "task_type": "knowledge",
            "model_type": "embeddings",
            "model_name": "knowledge_embeddings",
            "model_path": res.get("embeddings_path"),
            "n_docs": res.get("n_docs"),
            "trained_at": res.get("trained_at", None)
        }

    else:
        raise ValueError("Unknown task type for training")

    # register in local registry
    try:
        register_model_metadata(metadata)
    except Exception:
        pass

    # external registry if available
    if register_model_registry:
        try:
            register_model_registry(metadata)
        except Exception:
            pass

    # package model for download
    try:
        package_path = package_model_for_download(metadata)
        metadata["package"] = package_path
    except Exception:
        metadata["package"] = None

    return metadata
