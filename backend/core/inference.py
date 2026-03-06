import random

# --------------------------
# Classification Inference
# --------------------------
def classify_text(model, vectorizer, text: str) -> str:
    """
    Run inference for text classification
    """
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction


# --------------------------
# Chatbot Inference
# --------------------------
def chatbot_response(dataset, query: str) -> str:
    """
    Simple chatbot: match exact or closest question
    """
    for item in dataset:
        if query.lower() in item["question"].lower():
            return item["answer"]
    return random.choice([
        "I'm not sure, can you rephrase?",
        "Let me think... could you clarify?",
        "I don’t know that yet, but I’m learning!"
    ])


# --------------------------
# Knowledge Retrieval
# --------------------------
def knowledge_retrieval(files, query: str) -> str:
    """
    Mock retrieval: just list available files for now
    """
    return f"Searching knowledge docs for: {query}\nAvailable files: {files}"
