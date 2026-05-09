# 🫀 Heart Disease Prediction — End-to-End MLOps Project

> **Assignment:** MLOps Experimental Learning (S2-25_AMLCSZG523)  
> **Dataset:** Heart Disease UCI (Cleveland)  
> **Cloud Provider:** Google Cloud Platform (GKE)

---

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [EDA & Modelling](#eda--modelling)
5. [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
6. [Running the API Locally](#running-the-api-locally)
7. [Docker Containerization](#docker-containerization)
8. [CI/CD Pipeline (GitHub Actions → GKE)](#cicd-pipeline)
9. [Google Cloud Deployment (GKE)](#google-cloud-deployment-gke)
10. [Monitoring](#monitoring)
11. [API Reference](#api-reference)

---

## Architecture Overview

```
GitHub Push
    │
    ▼
GitHub Actions CI/CD
    ├── Lint (flake8 + black)
    ├── Unit Tests (pytest)
    ├── Train Model
    ├── Build Docker Image
    ├── Push → Google Container Registry (GCR)
    └── Deploy → Google Kubernetes Engine (GKE)
                    │
                    ▼
            LoadBalancer Service
                    │
                    ▼
          FastAPI (2 replicas + HPA)
          /predict  /health  /metrics
                    │
                    ▼
          Prometheus /metrics endpoint
```

---

## Project Structure

```
.
├── data/
│   └── processed.cleveland.data       # UCI Heart Disease dataset
├── models/                            # Saved model artifacts
│   ├── best_model.joblib
│   └── feature_cols.joblib
├── notebooks/
│   └── heart_disease_eda_training.ipynb  # EDA + Training notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py             # Data loading & preprocessing
│   ├── train.py                       # Model training + MLflow tracking
│   └── app.py                         # FastAPI serving application
├── tests/
│   ├── test_data_processing.py
│   └── test_app.py
├── deployment/
│   ├── deployment.yaml                # GKE Deployment manifest
│   ├── service.yaml                   # LoadBalancer Service
│   └── hpa.yaml                       # Horizontal Pod Autoscaler
├── .github/
│   └── workflows/
│       └── ci-cd.yml                  # GitHub Actions pipeline
├── monitoring/
│   ├── prometheus.yml                 # Prometheus scrape config
│   └── grafana-dashboard.json         # Grafana dashboard JSON
├── Dockerfile
├── .dockerignore
├── .gitignore
├── deploy.sh                          # One-click GKE deployment script
├── download_data.sh                   # Dataset download script
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Docker Desktop
- `gcloud` CLI (for GKE deployment)
- `kubectl`

### Local Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd mlopsi

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy dataset
# (Already done — data/processed.cleveland.data is included)
```

---

## EDA & Modelling

Open the notebook for full EDA and model training:

```bash
jupyter notebook notebooks/heart_disease_eda_training.ipynb
```

Or run training directly:

```bash
python -m src.train
```

**Models trained:**
| Model | ROC-AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.90 | ~0.84 | ~0.84 | ~0.85 |
| Random Forest | ~0.92 | ~0.86 | ~0.86 | ~0.87 |

---

## Experiment Tracking (MLflow)

```bash
mlflow ui --backend-store-uri mlruns --port 5000
# Open http://localhost:5000
```

All runs log: parameters, metrics, ROC-AUC, confusion matrix, and the full sklearn pipeline artifact.

---

## Running the API Locally

```bash
# Train model first
python -m src.train

# Start API
uvicorn src.app:app --host 0.0.0.0 --port 8080 --reload

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

**Interactive API docs:** http://localhost:8080/docs

---

## Docker Containerization

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container locally
docker run -p 8080:8080 heart-disease-api:latest

# Test
curl http://localhost:8080/health
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push to `main`:

| Step | Description |
|---|---|
| **Lint** | flake8 + black format check |
| **Test** | pytest with coverage report |
| **Train** | Train & save best model |
| **Build & Push** | Docker image → Google Container Registry |
| **Deploy** | kubectl apply → GKE cluster |

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `GCP_PROJECT_ID` | Your Google Cloud project ID |
| `GCP_SA_KEY` | Service account JSON key (base64) |

---

## Google Cloud Deployment (GKE)

### One-time Setup

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable \
  container.googleapis.com \
  containerregistry.googleapis.com

# Create GKE cluster
gcloud container clusters create heart-disease-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-standard-2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 4

# Get credentials
gcloud container clusters get-credentials heart-disease-cluster \
  --zone us-central1-a
```

### Build & Push Image

```bash
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE=gcr.io/$PROJECT_ID/heart-disease-api

gcloud auth configure-docker
docker build -t $IMAGE:latest .
docker push $IMAGE:latest
```

### Deploy to GKE

```bash
# Replace placeholder with your project ID
sed -i "s/YOUR_GCR_PROJECT_ID/$PROJECT_ID/g" deployment/deployment.yaml

kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/hpa.yaml

# Check rollout
kubectl rollout status deployment/heart-disease-api

# Get external IP
kubectl get svc heart-disease-api-service
```

Once the `EXTERNAL-IP` is assigned:
- **API:** `http://EXTERNAL_IP/predict`
- **Docs:** `http://EXTERNAL_IP/docs`
- **Health:** `http://EXTERNAL_IP/health`
- **Metrics:** `http://EXTERNAL_IP/metrics`

---

## Monitoring

The API exposes a `/metrics` endpoint in Prometheus format.

### Quick Prometheus + Grafana (local)

```bash
# Run Prometheus
docker run -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Run Grafana
docker run -p 3000:3000 grafana/grafana
```

**Key metrics tracked:**
- `api_requests_total` — total requests by endpoint and status
- `api_request_duration_seconds` — request latency histogram
- `predictions_total` — prediction count by class

---

## API Reference

### `POST /predict`

**Request body:**
```json
{
  "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
  "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
  "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "confidence": 0.8734,
  "probability_disease": 0.1266
}
```

### `GET /health`
```json
{"status": "ok"}
```

### `GET /metrics`
Prometheus metrics output.

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
