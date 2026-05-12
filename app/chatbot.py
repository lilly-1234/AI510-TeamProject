"""Helper functions for loading the trained model and predicting chatbot replies."""

import json
import os
from typing import Dict, Tuple

import joblib

# Paths for saved model and response mapping
MODEL_PATH = os.getenv("MODEL_PATH", "models/intent_model.joblib")
RESPONSES_PATH = os.getenv("RESPONSES_PATH", "models/response_map.json")

# Lower confidence threshold for better chatbot responses
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))


def load_artifacts():
    """Load trained model and response map from disk."""

    # Check whether trained model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Run python train_model.py first."
        )

    # Check whether response map exists
    if not os.path.exists(RESPONSES_PATH):
        raise FileNotFoundError(
            "Response map not found. Run python train_model.py first."
        )

    # Load trained ML model
    model = joblib.load(MODEL_PATH)

    # Load chatbot response mapping
    with open(RESPONSES_PATH, "r", encoding="utf-8") as file:
        response_map: Dict[str, str] = json.load(file)

    return model, response_map


# Load model and responses when application starts
model, response_map = load_artifacts()


def predict_intent(message: str) -> Tuple[str, float]:
    """Predict customer intent and confidence score."""

    # Get probability scores for all intents
    probabilities = model.predict_proba([message])[0]

    # Find highest probability
    best_index = probabilities.argmax()

    # Get predicted intent
    intent = model.classes_[best_index]

    # Get confidence score
    confidence = float(probabilities[best_index])

    return intent, confidence


def generate_reply(message: str) -> dict:
    """Generate chatbot response based on predicted intent."""

    # Predict intent from customer message
    intent, confidence = predict_intent(message)

    # Handle low-confidence predictions
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": "unknown",
            "confidence": round(confidence, 4),
            "response": (
                "I am sorry, I could not understand your request. "
                "Please contact a human support agent."
            ),
        }

    # Return predicted response
    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "response": response_map.get(
            intent,
            "A support agent will help you with this request."
        ),
    }