# tests/test_phase1.py
from core.prompt_parser import classify_prompt, train_ml_classifier

def test_rule_based():
    assert classify_prompt("Build a spam detection model") == "classification"
    assert classify_prompt("Create a chatbot for agriculture FAQ") == "chatbot"

def test_ml_fallback():
    vec, model = train_ml_classifier()
    result = classify_prompt("Dialogue system for restaurants", vec, model)
    assert result == "chatbot"
