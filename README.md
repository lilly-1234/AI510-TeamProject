# AI Customer Support Agent for E-Commerce using Intent Classification and MLOps

## Project Overview

This project presents an AI-based customer support agent for e-commerce platforms using Natural Language Processing (NLP) and Machine Learning. The chatbot automatically understands customer queries such as order tracking, refund requests, shipping information, payment issues, and account support using intent classification.

The system is built using:
- Python
- Flask API
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- MLOps concepts


# Features

- AI-based intent classification
- Customer support automation
- REST API using Flask
- Data preprocessing pipeline
- TF-IDF + Logistic Regression model
- Confidence-based predictions
- Metrics tracking
- Model registry logging
- Docker support

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Flask | REST API |
| Scikit-learn | Machine learning |
| Pandas | Data preprocessing |
| TF-IDF | Text vectorization |
| Logistic Regression | Intent classification |
| Joblib | Model saving/loading |
| Docker | Containerization |
| GitHub | Version control |

---

# Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd AI510-TeamProject
```

---

# Create Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Data Preprocessing

Run preprocessing script:

```bash
python preprocessing/data_cleaning.py
```

This step:
- removes missing values
- removes duplicate rows
- converts text to lowercase
- removes placeholders like {{Order Number}}
- saves cleaned dataset

Output:

```text
data/cleaned_customer_support_dataset.csv
```

---

# Model Training

Train the intent classification model:

```bash
python train_model.py --data data/cleaned_customer_support_dataset.csv
```

This step:
- trains the NLP model
- creates TF-IDF features
- performs intent classification
- saves trained model
- saves metrics
- creates model registry logs

Generated Files:

```text
models/intent_model.joblib
models/response_map.json
models/metrics.json
models/model_registry.log
```

---

# Run Flask API

Start the chatbot API:

```bash
python app/app.py
```

Server runs on:

```text
http://127.0.0.1:5000
```

---

python train_model.py --data data/cleaned_customer_support_dataset.csv

# API Endpoints

## Health Check

```http
GET /
```

Response:

```json
{
  "message": "AI Customer Support Agent API is running"
}
```

---

## Chat Endpoint

```http
POST /chat
```

Example Request:

```bash
curl -X POST http://127.0.0.1:5000/chat \
-H "Content-Type: application/json" \
-d '{"message":"I want to change my order."}'
```

Example Response:

```json
{
  "intent": "track_order",
  "confidence": 0.54,
  "response": "You can modify your order before it is shipped. Please provide your order number.",
  "timestamp": "2026-05-12T00:00:00Z"
}
```

---

# Example Test Messages

```text
I want to change my order.
need my refund.
My payment failed.
want to track my order

```
---

# Docker Support

Build Docker Image:

```bash
docker build -t ai-customer-support-agent .
```

Run Docker Container:

```bash
docker run -p 5000:5000 ai-customer-support-agent
```

---

# MLOps Features

This project includes:
- Data preprocessing pipeline
- Model training workflow
- Metrics logging
- Model registry tracking
- Deployment-ready API
- Docker containerization
- GitHub workflow support

# Future Improvements

- Deep Learning-based intent classification
- Cloud deployment using AWS
- Real-time chatbot UI
- Database integration
- Multi-language support
- Dynamic response generation using LLMs


