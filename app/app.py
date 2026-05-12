"""Flask API for the AI Customer Support Agent."""

from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

from chatbot import generate_reply

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"message": "AI Customer Support Agent API is running"})


@app.route("/chat", methods=["POST"])
def chat():
    """Receive customer message and return intent-based chatbot response."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    result = generate_reply(message)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return jsonify(result)


@app.route("/metadata", methods=["GET"])
def metadata():
    """Project metadata endpoint."""
    return jsonify(
        {
            "project": "AI Customer Support Agent for E-Commerce",
            "model": "TF-IDF + Logistic Regression Intent Classifier",
            "endpoints": ["GET /", "POST /chat", "GET /metadata"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
