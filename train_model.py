"""
Train an intent classification model for an AI customer support chatbot.

Expected dataset columns:
- instruction: customer question/text
- intent: intent label such as track_order, get_refund, payment_issue
- response: response for that intent
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Short demo-friendly responses for common intents
CUSTOM_RESPONSES = {
    "track_order": "Your order is currently being processed. You can check the order status using your order number.",
    "change_order": "You can modify your order before it is shipped. Please provide your order number.",
    "cancel_order": "Your order cancellation request has been received. Please provide your order number.",
    "get_refund": "Your refund request has been started. Refunds usually take 5-7 business days.",
    "payment_issue": "Please check your payment details and try again. If the issue continues, contact support.",
    "shipping_info": "Shipping usually takes 3-5 business days depending on your location.",
    "delivery_issue": "I am sorry about the delivery issue. Please provide your order number so we can check it.",
    "account_issue": "Please verify your account details. A support agent can help if the issue continues.",
    "contact_support": "A human support agent will help you with this request.",
    "delete_account": "I can help with account deletion. Please confirm your account details.",
}


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the CSV dataset."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"instruction", "intent", "response"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Keep only required columns and remove empty records
    df = df[["instruction", "intent", "response"]].dropna()

    # Clean text columns
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()

    # Remove empty rows
    df = df[(df["instruction"] != "") & (df["intent"] != "")]

    return df


def build_response_map(df: pd.DataFrame) -> dict:
    """Create one chatbot response for each intent."""

    response_map = {}

    for intent, group in df.groupby("intent"):
        # Use short custom response if available
        if intent in CUSTOM_RESPONSES:
            response_map[intent] = CUSTOM_RESPONSES[intent]
        else:
            # Otherwise use first dataset response
            response_map[intent] = group["response"].iloc[0]

    return response_map


def main() -> None:
    """Train model, save artifacts, and log metrics."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/customer_support.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Folder to save model files",
    )

    args = parser.parse_args()

    # Create model folder if it does not exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Load dataset
    df = load_dataset(args.data)

    X = df["instruction"]
    y = df["intent"]

    # Stratify only if every class has at least two examples
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    # TF-IDF converts text into numbers
    # Logistic Regression predicts the intent
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Test model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Create chatbot response map
    response_map = build_response_map(df)

    # Output paths
    model_path = os.path.join(args.model_dir, "intent_model.joblib")
    responses_path = os.path.join(args.model_dir, "response_map.json")
    metrics_path = os.path.join(args.model_dir, "metrics.json")
    registry_path = os.path.join(args.model_dir, "model_registry.log")

    # Save trained model
    joblib.dump(model, model_path)

    # Save response map
    with open(responses_path, "w", encoding="utf-8") as file:
        json.dump(response_map, file, indent=2)

    # Save model metrics
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": args.data,
        "rows": int(len(df)),
        "num_intents": int(df["intent"].nunique()),
        "accuracy": round(float(accuracy), 4),
        "report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Local model registry for MLOps tracking
    with open(registry_path, "a", encoding="utf-8") as file:
        file.write(
            f"{metrics['timestamp']} | "
            f"model=intent_model.joblib | "
            f"dataset={args.data} | "
            f"rows={metrics['rows']} | "
            f"intents={metrics['num_intents']} | "
            f"accuracy={metrics['accuracy']}\n"
        )

    print("Training completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"Responses saved to: {responses_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']}")


if __name__ == "__main__":
    main()