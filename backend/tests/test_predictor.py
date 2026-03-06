# tests/test_predictor.py
"""
Quick test runner for Predictor.
Usage:
  python -m tests.test_predictor --model path/to/model.joblib
"""

import argparse
from core.predictor import Predictor

def main(model_path):
    p = Predictor()
    meta = p.load(model_path)
    print("Loaded:", meta)
    print("Enter ':exit' to quit, ':info' to print loaded meta.")
    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q == ":exit":
            break
        if q == ":info":
            print(meta)
            continue
        res = p.predict_text(q)
        print("Result:", res)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True, help="Path to model/index/embeddings")
    args = ap.parse_args()
    main(args.model)
