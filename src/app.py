"""
FastAPI model serving application with request logging and Prometheus monitoring.
"""
import os
import time
import logging
import json
from datetime import datetime, timezone
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("heart-disease-api")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made", ["prediction"]
)
MODEL_LOADED = Gauge(
    "model_loaded", "Whether the model is loaded (1=yes, 0=no)"
)
ACTIVE_REQUESTS = Gauge(
    "active_requests", "Number of currently active requests"
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every incoming request with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next):
        ACTIVE_REQUESTS.inc()
        start = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            latency = time.time() - start
            status = response.status_code if response else 500
            endpoint = request.url.path
            method = request.method

            REQUEST_COUNT.labels(
                endpoint=endpoint, method=method, status=str(status)
            ).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
            ACTIVE_REQUESTS.dec()

            logger.info(
                f"{method} {endpoint} | status={status} | "
                f"latency={latency:.4f}s | client={request.client.host if request.client else 'unknown'}"
            )

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def load_model():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run training first."
        )
    return joblib.load(model_path)


app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment - Heart Disease Risk Classifier",
    version="1.0.0",
)
app.add_middleware(RequestLoggingMiddleware)

model = None


@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
        MODEL_LOADED.set(1)
        logger.info("✅ Model loaded successfully.")
    except FileNotFoundError as e:
        MODEL_LOADED.set(0)
        logger.warning(str(e))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    age: float = Field(..., example=63.0, description="Age in years")
    sex: float = Field(..., example=1.0, description="Sex (1=male, 0=female)")
    cp: float = Field(..., example=1.0, description="Chest pain type (1-4)")
    trestbps: float = Field(..., example=145.0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., example=233.0, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., example=1.0, description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: float = Field(..., example=2.0, description="Resting ECG results (0-2)")
    thalach: float = Field(..., example=150.0, description="Maximum heart rate achieved")
    exang: float = Field(..., example=0.0, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise")
    slope: float = Field(..., example=3.0, description="Slope of peak exercise ST segment")
    ca: float = Field(..., example=0.0, description="Number of major vessels colored by fluoroscopy")
    thal: float = Field(..., example=6.0, description="Thal: 3=normal, 6=fixed defect, 7=reversible defect")


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probability_disease: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"message": "Heart Disease Prediction API is running", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health_check():
    status = "ok" if model is not None else "model_not_loaded"
    return {"status": status}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: PatientFeatures, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        feature_values = np.array([[
            features.age, features.sex, features.cp, features.trestbps,
            features.chol, features.fbs, features.restecg, features.thalach,
            features.exang, features.oldpeak, features.slope, features.ca, features.thal
        ]])

        prediction = int(model.predict(feature_values)[0])
        probability = float(model.predict_proba(feature_values)[0][1])
        confidence = probability if prediction == 1 else 1 - probability
        label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        logger.info(
            f"PREDICTION | result={prediction} ({label}) | "
            f"prob={probability:.4f} | confidence={confidence:.4f} | "
            f"patient_age={features.age} sex={features.sex}"
        )

        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            confidence=round(confidence, 4),
            probability_disease=round(probability, 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8080, reload=False)
