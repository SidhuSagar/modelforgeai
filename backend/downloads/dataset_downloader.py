import os
import requests
import kaggle

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

# Kaggle datasets
KAGGLE_DATASETS = {
    "classification": {
        "sms_spam.csv": "uciml/sms-spam-collection-dataset",
        "imdb_sentiment.csv": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "news_topics.csv": "amananandrai/ag-news-classification-dataset",
    }
}

# Non-Kaggle datasets
NON_KAGGLE_DATASETS = {
    "chatbot": {
        "faq.yml": "https://raw.githubusercontent.com/sdushantha/sample-yaml/master/example.yml",
        "agriculture_faq.yml": "https://raw.githubusercontent.com/deepmipt/DeepPavlov/master/deeppavlov/configs/faq/faq_agriculture.json"
    },
    "knowledge": {
        "agriculture.pdf": "https://arxiv.org/pdf/2402.15351.pdf",
        "sample.txt": "https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt"
    }
}

def make_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    for folder in ["classification", "chatbot", "knowledge"]:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

def download_non_kaggle():
    for folder, files in NON_KAGGLE_DATASETS.items():
        for fname, url in files.items():
            save_path = os.path.join(BASE_DIR, folder, fname)
            if not os.path.exists(save_path):
                print(f"Downloading {fname} ...")
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    r.raise_for_status()
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"[OK] Saved: {save_path}")
                except Exception as e:
                    print(f"[ERR] Failed {url} -> {e}")
            else:
                print(f"[SKIP] Already exists: {save_path}")

def download_kaggle():
    for fname, dataset in KAGGLE_DATASETS["classification"].items():
        save_path = os.path.join(BASE_DIR, "classification", fname)
        if not os.path.exists(save_path):
            print(f"Downloading {fname} from Kaggle ...")
            os.system(f"kaggle datasets download -d {dataset} -p {os.path.join(BASE_DIR, 'classification')} --unzip")
        else:
            print(f"[SKIP] Already exists: {save_path}")

if __name__ == "__main__":
    make_dirs()
    download_kaggle()
    download_non_kaggle()
    print("✅ All Phase 2 datasets downloaded and stored!")
