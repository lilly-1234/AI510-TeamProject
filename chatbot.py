"""Helper functions for loading the trained model and predicting chatbot replies."""

import json
import os
from typing import Dict, Tuple

import joblib

MODEL_PATH = os.getenv("MODEL_PATH", "models/intent_model.joblib")
RESPONSES_PATH = os.getenv("RESPONSES_PATH", "models/response_map.json")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))


def load_artifacts():
    """Load model and response map from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run python train_model.py first.")
    if not os.path.exists(RESPONSES_PATH):
        raise FileNotFoundError("Response map not found. Run python train_model.py first.")

    model = joblib.load(MODEL_PATH)
    with open(RESPONSES_PATH, "r", encoding="utf-8") as file:
        response_map: Dict[str, str] = json.load(file)
    return model, response_map


model, response_map = load_artifacts()


def predict_intent(message: str) -> Tuple[str, float]:
    """Predict intent and confidence score for the customer message."""
    probabilities = model.predict_proba([message])[0]
    best_index = probabilities.argmax()
    intent = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    return intent, confidence


def generate_reply(message: str) -> dict:
    """Return chatbot response with intent and confidence."""
    intent, confidence = predict_intent(message)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": "unknown",
            "confidence": round(confidence, 4),
            "response": "I am sorry, I could not understand your request. Please contact a human support agent.",
        }

    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "response": response_map.get(intent, "A support agent will help you with this request."),
    }
