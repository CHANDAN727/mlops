"""
Unit tests for the FastAPI prediction app.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- Test prediction logic (mocked model) -----------------------------------
class MockModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_mock_model_predict():
    model = MockModel()
    X = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    pred = model.predict(X)
    assert pred[0] in [0, 1]


def test_mock_model_predict_proba():
    model = MockModel()
    X = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6


def test_mock_model_confidence():
    model = MockModel()
    X = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    confidence = probability if prediction == 1 else 1 - probability
    assert 0.0 <= confidence <= 1.0


# ---- Test API schemas --------------------------------------------------------
def test_patient_features_schema():
    from src.app import PatientFeatures
    features = PatientFeatures(
        age=63, sex=1, cp=1, trestbps=145, chol=233,
        fbs=1, restecg=2, thalach=150, exang=0,
        oldpeak=2.3, slope=3, ca=0, thal=6
    )
    assert features.age == 63
    assert features.sex == 1


def test_prediction_label_disease():
    prediction = 1
    label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    assert label == "Heart Disease Detected"


def test_prediction_label_no_disease():
    prediction = 0
    label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    assert label == "No Heart Disease"
