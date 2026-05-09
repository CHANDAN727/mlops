"""
Unit tests for model training module.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import joblib  # noqa: E402

from src.data_processing import (  # noqa: E402
    load_data, preprocess, build_preprocessing_pipeline,
    get_train_test_split,
)
from src.train import build_model_pipeline, evaluate_model  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed.cleveland.data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_model.joblib")


@pytest.fixture(scope="module")
def data():
    df = load_data(DATA_PATH)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    return X_train, X_test, y_train, y_test


def test_logistic_regression_pipeline(data):
    X_train, X_test, y_train, y_test = data
    preprocessing = build_preprocessing_pipeline()
    model = LogisticRegression(max_iter=500, random_state=42)
    pipeline = build_model_pipeline(model, preprocessing)
    pipeline.fit(X_train, y_train)
    metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
    assert metrics["accuracy"] > 0.7, "LR accuracy must be > 70%"
    assert metrics["roc_auc"] > 0.7, "LR ROC-AUC must be > 70%"
    assert len(y_pred) == len(y_test)


def test_random_forest_pipeline(data):
    X_train, X_test, y_train, y_test = data
    preprocessing = build_preprocessing_pipeline()
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    pipeline = build_model_pipeline(model, preprocessing)
    pipeline.fit(X_train, y_train)
    metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
    assert metrics["accuracy"] > 0.7, "RF accuracy must be > 70%"
    assert metrics["roc_auc"] > 0.7, "RF ROC-AUC must be > 70%"


def test_evaluate_model_returns_all_metrics(data):
    X_train, X_test, y_train, y_test = data
    preprocessing = build_preprocessing_pipeline()
    model = LogisticRegression(max_iter=500, random_state=42)
    pipeline = build_model_pipeline(model, preprocessing)
    pipeline.fit(X_train, y_train)
    metrics, _ = evaluate_model(pipeline, X_test, y_test)
    for key in ["accuracy", "precision", "recall", "roc_auc"]:
        assert key in metrics, f"Missing metric: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range"


def test_saved_model_loads_and_predicts():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not trained yet")
    pipeline = joblib.load(MODEL_PATH)
    sample = np.array([[63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0]])
    pred = pipeline.predict(sample)
    proba = pipeline.predict_proba(sample)
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6


def test_pipeline_has_correct_steps(data):
    X_train, _, y_train, _ = data
    preprocessing = build_preprocessing_pipeline()
    model = LogisticRegression(max_iter=500, random_state=42)
    pipeline = build_model_pipeline(model, preprocessing)
    pipeline.fit(X_train, y_train)
    assert "preprocessing" in pipeline.named_steps
    assert "model" in pipeline.named_steps
