import os
import pandas as pd
import yaml

# Base dataset directory
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# -------------------- Classification --------------------
def load_classification_dataset(name="sms_spam"):
    """
    Load and clean classification datasets.
    """
    path = os.path.join(DATASET_DIR, "classification")
    if name == "sms_spam":
        df = pd.read_csv(os.path.join(path, "spam.csv"), encoding="latin-1")

        # Keep only required columns
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]

        # Save cleaned version
        save_path = os.path.join(PROCESSED_DIR, "sms_spam_clean.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ Cleaned dataset saved at {save_path}")
        return df
    else:
        raise ValueError("❌ Unknown classification dataset name")


# -------------------- Chatbot --------------------
def load_chatbot_dataset(name="faq"):
    """
    Load chatbot dataset (YAML format).
    """
    path = os.path.join(DATASET_DIR, "chatbot", f"{name}.yml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Chatbot dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Save processed version
    save_path = os.path.join(PROCESSED_DIR, f"{name}_clean.yml")
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    print(f"✅ Cleaned chatbot dataset saved at {save_path}")

    return data


# -------------------- Knowledge --------------------
def load_knowledge_dataset():
    """
    Load all text/pdf files inside datasets/knowledge/
    """
    path = os.path.join(DATASET_DIR, "knowledge")
    if not os.path.exists(path):
        raise FileNotFoundError("❌ Knowledge folder not found.")

    files = [
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if fname.endswith((".pdf", ".txt"))
    ]
    return files


# -------------------- Main Loader --------------------
def load_dataset(task_type, dataset_name=None):
    """
    Unified dataset loader for different task types.
    """
    if task_type == "classification":
        return load_classification_dataset(name=dataset_name)
    elif task_type == "chatbot":
        return load_chatbot_dataset(name=dataset_name)
    elif task_type == "knowledge":
        return load_knowledge_dataset()
    else:
        raise ValueError("❌ Unknown task type")
