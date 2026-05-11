# AI Customer Support Agent for E-Commerce using Intent Classification and MLOps

This project builds an AI customer support chatbot for e-commerce. It classifies customer messages into intents such as order tracking, refunds, delivery, payment issues, account support, and human-agent contact. After predicting the intent, the system returns a suitable support response.

## Project Features

- Intent classification using TF-IDF and Logistic Regression
- Flask REST API with `/chat` endpoint
- Simple HTML chatbot UI
- Model metrics and local model registry log
- Docker support
- GitHub Actions CI pipeline
- Unit tests using pytest

## Project Structure

```text
ai_customer_support_agent/
├── app/
│   ├── app.py
│   ├── chatbot.py
│   └── ui.html
├── data/
│   └── sample_customer_support.csv
├── models/
│   ├── intent_model.joblib
│   ├── response_map.json
│   ├── metrics.json
│   └── model_registry.log
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── README.md
├── requirements.txt
└── train_model.py
```

## Dataset

Recommended dataset: Bitext Gen AI Customer Support Dataset from Kaggle.

Expected CSV columns:

```text
instruction,intent,response
```

Place the downloaded dataset here:

```text
data/customer_support.csv
```

A small sample dataset is already included for quick testing.

## Step 1: Create Virtual Environment

### Windows PowerShell

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

## Step 3: Train the Model

Using sample data:

```bash
python train_model.py --data data/sample_customer_support.csv
```

Using Kaggle dataset:

```bash
python train_model.py --data data/customer_support.csv
```

After training, these files are created in the `models` folder:

```text
intent_model.joblib
response_map.json
metrics.json
model_registry.log
```

## Step 4: Run Flask API

```bash
python app/app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## Step 5: Test the Chat API

### Windows PowerShell

```bash
curl.exe -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d "{\"message\":\"Where is my order?\"}"
```

### macOS/Linux/Git Bash

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Where is my order?"}'
```

Expected output:

```json
{
  "intent": "track_order",
  "confidence": 0.65,
  "response": "Please provide your order number so I can help track your order.",
  "timestamp": "2026-05-11T00:00:00Z"
}
```

## Step 6: Run Simple UI

Open this file in the browser:

```text
app/ui.html
```

Make sure Flask is already running before using the UI.

## Step 7: Run Tests

```bash
pytest -q
```

## Step 8: Run with Docker

Build image:

```bash
docker build -t ai-customer-support-agent .
```

Run container:

```bash
docker run -p 5000:5000 ai-customer-support-agent
```

Test:

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"I need a refund"}'
```

## Sample Test Prompts

Use these prompts to test different scenarios:

```text
Where is my order?
I want to cancel my order.
How do I get a refund?
My payment failed.
What payment methods do you accept?
How long does delivery take?
I forgot my password.
I want to talk to a human agent.
I need to change my shipping address.
```

## MLOps Part

This project includes basic MLOps practices:

- `model_registry.log` stores model name, dataset, timestamp, rows, intents, and accuracy.
- `metrics.json` stores model evaluation results.
- `Dockerfile` containerizes the application.
- `ci.yml` trains the model and runs tests automatically in GitHub Actions.

## Git Commands

```bash
git init
git add .
git commit -m "Initial AI customer support agent project"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```
