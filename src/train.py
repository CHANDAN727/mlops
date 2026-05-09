"""
Model training module with MLflow experiment tracking.
Logs parameters, metrics, artifacts (model + plots) for every run.
"""
import os
import logging
import tempfile
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from src.data_processing import (
    load_data, preprocess, build_preprocessing_pipeline, get_train_test_split, FEATURE_COLS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODELS_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "heart-disease-classification"


def build_model_pipeline(model, preprocessing_pipeline):
    """Build full pipeline: preprocessing + model."""
    return Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", model),
    ])


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate a trained pipeline on test data."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps["model"], "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }
    return metrics, y_pred


def train_and_log(model_name, model, X_train, X_test, y_train, y_test, params):
    """Train a model and log everything to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        # Build pipeline
        preprocessing = build_preprocessing_pipeline()
        pipeline = build_model_pipeline(model, preprocessing)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        logger.info(f"{model_name} CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
        metrics["cv_roc_auc_mean"] = cv_scores.mean()
        metrics["cv_roc_auc_std"] = cv_scores.std()

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # --- Log plots as artifacts ---
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Confusion Matrix
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred,
                display_labels=["No Disease", "Disease"],
                cmap="Blues", ax=ax_cm, colorbar=False,
            )
            ax_cm.set_title(f"{model_name} — Confusion Matrix")
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig_cm)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # ROC Curve
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc, name=model_name)
            ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax_roc.set_title(f"{model_name} — ROC Curve")
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close(fig_roc)
            mlflow.log_artifact(roc_path, artifact_path="plots")

            # Classification report as text
            report = classification_report(y_test, y_pred)
            report_path = os.path.join(tmpdir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(f"{model_name} — Classification Report\n")
                f.write("=" * 50 + "\n")
                f.write(report)
            mlflow.log_artifact(report_path, artifact_path="reports")

        logger.info(f"{model_name} Test Metrics: {metrics}")
        print(f"\n{model_name} - Classification Report:")
        print(classification_report(y_test, y_pred))

        return pipeline, metrics


def train_all(data_path: str):
    """Train all models and save the best one."""
    df = load_data(data_path)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    results = {}

    # --- Logistic Regression ---
    lr_params = {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "random_state": 42}
    lr_model = LogisticRegression(**lr_params)
    lr_pipeline, lr_metrics = train_and_log(
        "LogisticRegression", lr_model, X_train, X_test, y_train, y_test, lr_params
    )
    results["LogisticRegression"] = (lr_pipeline, lr_metrics)

    # --- Random Forest ---
    rf_params = {"n_estimators": 200, "max_depth": 8, "min_samples_split": 5, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)
    rf_pipeline, rf_metrics = train_and_log(
        "RandomForest", rf_model, X_train, X_test, y_train, y_test, rf_params
    )
    results["RandomForest"] = (rf_pipeline, rf_metrics)

    # --- Select best model ---
    best_name = max(results, key=lambda k: results[k][1].get("roc_auc", 0))
    best_pipeline, best_metrics = results[best_name]
    logger.info(f"\nBest model: {best_name} with ROC-AUC={best_metrics.get('roc_auc', 0):.4f}")

    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Best model saved to {model_path}")

    # Save feature column names for inference
    feature_path = os.path.join(MODELS_DIR, "feature_cols.joblib")
    joblib.dump(FEATURE_COLS, feature_path)

    return best_pipeline, best_name, best_metrics


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed.cleveland.data")
    train_all(data_path)
