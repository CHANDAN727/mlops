"""
Data processing module for Heart Disease UCI Dataset.
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_data(data_path: str) -> pd.DataFrame:
    """Load the Cleveland heart disease dataset."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, header=None, names=COLUMNS, na_values="?")
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    df = df.copy()
    # Binarize target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)
    # Drop rows where target is NaN
    df = df.dropna(subset=["target"])
    logger.info(f"After preprocessing: {len(df)} rows")
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """Build a scikit-learn preprocessing pipeline."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train/test sets."""
    X = df[FEATURE_COLS].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed.cleveland.data")
    df = load_data(data_path)
    df = preprocess(df)
    print(df.head())
    print(df["target"].value_counts())
