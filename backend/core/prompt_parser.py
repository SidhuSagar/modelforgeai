# core/prompt_parser.py
"""
Prompt parsing / task detection helpers.

Provides:
 - classify_prompt_rule(prompt)        : rule-based
 - train_ml_classifier()               : returns (vectorizer, model)
 - classify_prompt_ml(prompt, vec, mdl): ML prediction
 - classify_prompt(prompt, vec=None, mdl=None): hybrid rule->ML
 - get_task_type(user_choice=None, user_prompt=None): unified selector
"""

import re
from typing import Tuple, Optional

# ----------------------------
# RULE-BASED CLASSIFIER
# ----------------------------
def classify_prompt_rule(prompt: str) -> str:
    """
    Rule-based classification into 'classification', 'chatbot', or 'unknown'.
    """
    if not prompt:
        return "unknown"

    text = prompt.lower()

    classification_keywords = [
        "classify", "sentiment", "spam", "detection", "analyze", "topic modeling",
        "intent recognition", "categorize", "language identification", "text mining",
        "tagging", "predict label", "supervised learning", "sort emails", "document classification",
        "predict", "labels", "label", "classifier", "classification"
    ]

    chatbot_keywords = [
        "chatbot", "conversation", "dialogue", "faq", "support", "assistant",
        "talk", "customer service", "helpdesk", "qna", "virtual agent",
        "interactive", "gpt", "bot", "response generator", "chat application",
        "reply", "answer", "question", "chat"
    ]

    classification_patterns = [r"\bclassif\w*", r"\bdetect\w*", r"\bpredict\w*", r"\btag\w*"]
    chatbot_patterns = [r"\bchat\w*", r"\btalk\w*", r"\bbot\w*", r"\bconvers\w*", r"\bassist\w*"]

    scores = {"classification": 0, "chatbot": 0}

    for word in classification_keywords:
        if word in text:
            scores["classification"] += 1
    for word in chatbot_keywords:
        if word in text:
            scores["chatbot"] += 1

    for pattern in classification_patterns:
        if re.search(pattern, text):
            scores["classification"] += 1
    for pattern in chatbot_patterns:
        if re.search(pattern, text):
            scores["chatbot"] += 1

    if scores["classification"] > scores["chatbot"]:
        return "classification"
    elif scores["chatbot"] > scores["classification"]:
        return "chatbot"
    else:
        return "unknown"


# ----------------------------
# ML-BASED CLASSIFIER
# ----------------------------
def train_ml_classifier() -> Tuple:
    """
    Train a simple ML classifier for prompt type detection.
    Returns (vectorizer, model).

    NOTE: lightweight training on a few examples; adequate for prompt routing but not
    production-grade. Called repeatedly is cheap for small dataset here.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise ImportError("scikit-learn required for train_ml_classifier. Install scikit-learn.") from e

    X_train = [
        "Build a spam detection model",
        "Make a chatbot for customer support",
        "Sentiment analysis on tweets",
        "FAQ answering bot",
        "Categorize documents into topics",
        "AI assistant for QnA",
        "Predict email intent",
        "Dialogue system for restaurants",
        "Train a classifier to tag news articles",
        "Create a conversational agent",
        "Index documents for knowledge base",
        "Search and answer from PDF documents"
    ]
    y_train = [
        "classification", "chatbot", "classification", "chatbot",
        "classification", "chatbot", "classification", "chatbot",
        "classification", "chatbot", "knowledge", "knowledge"
    ]

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return vectorizer, model


def classify_prompt_ml(prompt: str, vectorizer, model) -> str:
    """
    Predict prompt type using trained ML classifier.
    """
    if not prompt:
        return "unknown"
    X_test = vectorizer.transform([prompt])
    pred = model.predict(X_test)[0]
    return pred


# ----------------------------
# HYBRID CLASSIFIER (RULE + ML)
# ----------------------------
def classify_prompt(prompt: str, vectorizer=None, model=None) -> str:
    """
    Hybrid classifier: rule-based first, ML fallback if unknown.
    Keeps the same signature as previous code (backwards compatible).
    """
    rule_result = classify_prompt_rule(prompt)
    if rule_result != "unknown":
        return rule_result

    if vectorizer is None or model is None:
        vectorizer, model = train_ml_classifier()
    try:
        return classify_prompt_ml(prompt, vectorizer, model)
    except Exception:
        return "unknown"


# ----------------------------
# UNIFIED SELECTOR FOR UI/BACKEND
# ----------------------------
def get_task_type(user_choice: Optional[str] = None, user_prompt: Optional[str] = None) -> str:
    """
    Unified task selector: supports both manual selection and auto-detection from prompt.

    Args:
        user_choice: manually selected task (classification/chatbot/knowledge)
        user_prompt: prompt text for auto detection

    Returns:
        'classification' | 'chatbot' | 'knowledge' | 'unknown'
    """
    valid_tasks = ["classification", "chatbot", "knowledge"]

    # 1) Manual selection (explicit priority)
    if user_choice:
        choice = user_choice.strip().lower()
        if choice in valid_tasks:
            return choice

    # 2) Auto-detection via prompt (hybrid rule + ML)
    if user_prompt and user_prompt.strip():
        # reuse local functions - no circular imports
        try:
            # Try hybrid classify_prompt first
            vectorizer, model = None, None
            # Train small classifier once (cheap)
            vectorizer, model = train_ml_classifier()
            detected = classify_prompt(user_prompt, vectorizer, model)
            # If ML returns 'knowledge' or rule detection yields 'unknown' fallback to 'unknown'
            if detected in valid_tasks:
                return detected
        except Exception:
            pass

    # fallback
    return "unknown"
