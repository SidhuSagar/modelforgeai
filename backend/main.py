"""
Model Forge AI - Main pipeline runner

Combines:
- Task selection (classification / chatbot / knowledge)
- Dataset loading
- Preprocessing config
- Training / index building
- Visualization (classification)
- Packaging & metadata registration
- Interactive testing (optional)
"""

import os
import json
import argparse
from datetime import datetime

# Core imports
from core.prompt_parser import get_task_type
from core.dataset_handler import load_dataset
from core.model_trainer import (
    train_classification,
    build_chatbot_index,
    build_knowledge_embeddings,
)
from core.model_manager import (
    register_model_metadata,
    package_model_for_download,
)
from core.visualizer import visualize_training_summary
from core.predictor import Predictor

DEFAULT_LOG_PATH = "predictions.log"


# ─────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────

def safe_print_json(obj):
    """Pretty-print JSON safely."""
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(str(obj))


def append_prediction_log(log_path, sample, prediction):
    """Append prediction results to a log file."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sample": sample,
                "prediction": prediction,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write prediction log: {e}")


# ─────────────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Model Forge AI - Pipeline Runner")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive testing phase",
    )
    parser.add_argument(
        "--log-predictions",
        action="store_true",
        help="Append interactive predictions to predictions.log",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────

def run_pipeline(no_interactive=False, log_predictions=False):
    os.system("cls" if os.name == "nt" else "clear")
    print("🧠 MODEL FORGE AI – Full Intelligent Runner")
    print("===========================================\n")

    # Step 1: Task Selection
    print("Step 1️⃣ : Select your task type")
    print("Options: [classification] | [chatbot] | [knowledge]")
    manual_choice = input("👉 Enter task type (or Press Enter for Auto-detect): ").strip().lower()

    auto_prompt = None
    if not manual_choice:
        auto_prompt = input("💬 Describe your project briefly (AI will detect task): ")

    task_type = get_task_type(user_choice=manual_choice, user_prompt=auto_prompt)
    print(f"✅ Task selected: {task_type}\n")

    # Step 2: Model Type
    print("Step 2️⃣ : Choose your model type")
    print("Options: [auto] | [logisticregression] | [randomforest] | [svm] | [llm]")
    model_type = input("👉 Enter model type (default auto): ").strip().lower() or "auto"
    print(f"✅ Model type selected: {model_type}\n")

    # Step 3: Dataset Selection
    print("Step 3️⃣ : Select Dataset")
    custom_path = input("📂 Enter dataset path (or press Enter to use built-in): ").strip() or None

    dataset_name = None
    if not custom_path:
        if task_type == "classification":
            dataset_name = input("Built-in dataset [sms_spam]: ").strip() or "sms_spam"
        elif task_type == "chatbot":
            dataset_name = input("Built-in dataset [faq]: ").strip() or "faq"
        else:
            print("📄 Using all available knowledge documents.")

    dataset = load_dataset(task_type, dataset_name=dataset_name, custom_path=custom_path)
    print("✅ Dataset loaded successfully.\n")

    # Step 4: Preprocessing
    print("Step 4️⃣ : Preprocessing Setup")
    test_size_input = input("📊 Test split ratio (default 0.2): ").strip()
    try:
        test_size = float(test_size_input) if test_size_input else 0.2
    except ValueError:
        test_size = 0.2
        print("⚠️ Invalid input — using default 0.2")

    # Step 5: Training Configuration
    print("\nStep 5️⃣ : Training Setup")
    epochs_input = input("🔁 Epochs (LLM only, default 3): ").strip()
    try:
        epochs = int(epochs_input) if epochs_input else 3
    except ValueError:
        epochs = 3
        print("⚠️ Invalid input — using default 3")

    # Step 6: Training
    print("\n🚀 Starting Training...\n")
    try:
        if task_type == "classification":
            model_info = train_classification(
                dataset=dataset,
                model_type=model_type,
                test_size=test_size,
                hyperparams={"epochs": epochs} if model_type == "llm" else {},
            )
        elif task_type == "chatbot":
            model_info = build_chatbot_index(dataset_name or "faq")
        elif task_type == "knowledge":
            model_info = build_knowledge_embeddings()
        else:
            print("❌ Invalid task type")
            return
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    print("✅ Training Complete!\n")
    safe_print_json(model_info)

    # Step 7: Visualization
    if task_type == "classification" and isinstance(model_info, dict) and "report" in model_info:
        try:
            vis_path = visualize_training_summary(model_info)
            print(f"📊 Visualization saved at: {vis_path}")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

    # Step 8: Packaging
    if isinstance(model_info, dict) and "model_path" in model_info:
        try:
            register_model_metadata(model_info)
            zip_path = package_model_for_download(model_info)
            print(f"📦 Packaged model available at:\n{zip_path}")
        except Exception as e:
            print(f"⚠️ Packaging failed: {e}")

    # Step 9: Interactive Testing
    if no_interactive:
        print("🔕 Interactive testing skipped.")
        return

    print("\n🧪 Test Your Model")
    tester = Predictor()

    if isinstance(model_info, dict) and "model_path" in model_info:
        try:
            tester.load(model_info["model_path"])
            print("✅ Model loaded for prediction.\n")
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return
    else:
        print("⚠️ No model available for testing.")
        return

    print("Type text to predict | :info | :exit\n")
    while True:
        sample = input("> ").strip()
        if sample == ":exit":
            break
        if sample == ":info":
            safe_print_json(model_info)
            continue
        if not sample:
            continue
        try:
            prediction = tester.predict_text(sample)
            safe_print_json(prediction)
            if log_predictions:
                append_prediction_log(DEFAULT_LOG_PATH, sample, prediction)
        except Exception as e:
            print(f"Prediction error: {e}")

    print("\n🎉 DONE — Model Forge AI Training Complete!\n")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        no_interactive=args.no_interactive,
        log_predictions=args.log_predictions,
    )
