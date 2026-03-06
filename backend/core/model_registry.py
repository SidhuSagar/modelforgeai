# core/model_registry.py
import os
import json

REGISTRY_PATH = os.path.join("outputs", "registry.json")
os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

def register_model(model_info):
    """Save model metadata to registry file."""
    registry = []
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            try:
                registry = json.load(f)
            except json.JSONDecodeError:
                registry = []

    registry.append(model_info)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def list_registry():
    """List all registered models."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
