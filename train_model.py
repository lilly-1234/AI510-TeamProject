"""
Train an intent classification model for an AI customer support chatbot.

Expected dataset columns:
- instruction: customer question/text
- intent: intent label such as track_order, get_refund, payment_issue
- response: sample response for that intent

Run:
    python train_model.py --data data/customer_support.csv
If you do not have the Kaggle dataset yet, run:
    python train_model.py --data data/sample_customer_support.csv
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


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = {"instruction", "intent", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Keep only required columns and remove empty records.
    df = df[["instruction", "intent", "response"]].dropna()
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    df = df[(df["instruction"] != "") & (df["intent"] != "")]
    return df


def build_response_map(df: pd.DataFrame) -> dict:
    """Create one default response for each intent."""
    response_map = {}
    for intent, group in df.groupby("intent"):
        # Pick the first response for each intent.
        response_map[intent] = group["response"].iloc[0]
    return response_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/customer_support.csv", help="Path to CSV dataset")
    parser.add_argument("--model-dir", default="models", help="Folder to save model files")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    df = load_dataset(args.data)

    X = df["instruction"]
    y = df["intent"]

    # Stratify only if every class has at least two examples.
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # TF-IDF converts text into numeric features. Logistic Regression predicts intent.
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    response_map = build_response_map(df)

    model_path = os.path.join(args.model_dir, "intent_model.joblib")
    responses_path = os.path.join(args.model_dir, "response_map.json")
    metrics_path = os.path.join(args.model_dir, "metrics.json")
    registry_path = os.path.join(args.model_dir, "model_registry.log")

    joblib.dump(model, model_path)
    with open(responses_path, "w", encoding="utf-8") as file:
        json.dump(response_map, file, indent=2)

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": args.data,
        "rows": int(len(df)),
        "num_intents": int(df["intent"].nunique()),
        "accuracy": round(float(accuracy), 4),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Simple local model registry for MLOps tracking.
    with open(registry_path, "a", encoding="utf-8") as file:
        file.write(
            f"{metrics['timestamp']} | model=intent_model.joblib | "
            f"dataset={args.data} | rows={metrics['rows']} | "
            f"intents={metrics['num_intents']} | accuracy={metrics['accuracy']}\n"
        )

    print("Training completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"Accuracy: {metrics['accuracy']}")


if __name__ == "__main__":
    main()
