"""
Unit tests for data processing module.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from src.data_processing import (
    load_data, preprocess, build_preprocessing_pipeline,
    get_train_test_split, COLUMNS, FEATURE_COLS
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed.cleveland.data")


@pytest.fixture
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture
def processed_df(raw_df):
    return preprocess(raw_df)


def test_load_data_shape(raw_df):
    assert raw_df.shape[1] == 14, "Dataset must have 14 columns"
    assert len(raw_df) > 0, "Dataset must not be empty"


def test_load_data_columns(raw_df):
    assert list(raw_df.columns) == COLUMNS


def test_preprocess_target_binary(processed_df):
    unique_targets = processed_df["target"].unique()
    assert set(unique_targets).issubset({0, 1}), "Target must be binary (0 or 1)"


def test_preprocess_no_nan_target(processed_df):
    assert processed_df["target"].isna().sum() == 0, "Target column must have no NaN"


def test_preprocessing_pipeline_fit_transform(processed_df):
    X = processed_df[FEATURE_COLS].values
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert X_transformed.shape == X.shape, "Transformed shape must match input shape"
    # After scaling, mean should be ~0
    assert abs(X_transformed.mean()) < 0.5


def test_train_test_split_sizes(processed_df):
    X_train, X_test, y_train, y_test = get_train_test_split(processed_df, test_size=0.2)
    total = len(processed_df.dropna(subset=["target"]))
    assert len(X_train) + len(X_test) == total
    assert len(y_train) + len(y_test) == total


def test_train_test_split_stratified(processed_df):
    X_train, X_test, y_train, y_test = get_train_test_split(processed_df, test_size=0.2)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.1, "Stratification should maintain class ratio"


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 13, "Must have 13 feature columns"
