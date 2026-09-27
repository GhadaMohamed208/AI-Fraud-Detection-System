"""
Automated tests for the fraud-detection preprocessing pipeline.
Run with:  pytest tests/test_preprocessing.py -v
Assumes the notebook has already been run once so that data/processed/*.csv and
artifacts/* exist under the project root.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import FraudPreprocessor  # noqa: E402


@pytest.fixture(scope="module")
def feature_config():
    with open(os.path.join(ARTIFACTS_DIR, "feature_config.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline():
    return FraudPreprocessor.load(os.path.join(ARTIFACTS_DIR, "preprocessing_pipeline.pkl"))


@pytest.fixture(scope="module")
def splits():
    train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "validation.csv"))
    test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv"))
    return train, val, test


def test_raw_data_can_be_loaded():
    # If we got this far via the fixtures, processed data exists, which implies the
    # raw data was loadable when the notebook ran.
    assert os.path.exists(DATA_PROCESSED_DIR)


def test_target_column_exists_in_processed_splits(splits):
    train, val, test = splits
    for df in (train, val, test):
        assert "Class" in df.columns


def test_target_not_in_feature_order(feature_config):
    assert feature_config["target"] not in feature_config["feature_order"]


def test_feature_columns_match_across_splits(splits, feature_config):
    train, val, test = splits
    feat_cols = feature_config["feature_order"]
    for df in (train, val, test):
        assert list(df.columns[:-1]) == feat_cols  # last column is Class
        assert set(feat_cols).issubset(set(df.columns))


def test_feature_order_identical_across_splits(splits):
    train, val, test = splits
    train_feats = [c for c in train.columns if c != "Class"]
    val_feats = [c for c in val.columns if c != "Class"]
    test_feats = [c for c in test.columns if c != "Class"]
    assert train_feats == val_feats == test_feats


def test_pipeline_can_be_loaded(pipeline):
    assert pipeline is not None
    assert pipeline._is_fit is True


def test_sample_transaction_can_be_transformed(pipeline, feature_config):
    raw_sample = pd.DataFrame([{
        "Time": 0.0, "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }])
    out = pipeline.transform(raw_sample)
    assert out.shape[0] == 1
    assert list(out.columns) == feature_config["feature_order"]


def test_transformed_output_has_expected_feature_count(pipeline, feature_config):
    raw_sample = pd.DataFrame([{
        "Time": 0.0, "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }])
    out = pipeline.transform(raw_sample)
    assert out.shape[1] == feature_config["feature_count"]


def test_no_nan_or_inf_in_processed_splits(splits, feature_config):
    train, val, test = splits
    feat_cols = feature_config["feature_order"]
    for df in (train, val, test):
        assert not df[feat_cols].isna().any().any()
        assert not np.isinf(df[feat_cols].to_numpy(dtype=float)).any()


def test_split_sizes_are_reasonable(splits):
    train, val, test = splits
    total = len(train) + len(val) + len(test)
    train_frac = len(train) / total
    val_frac = len(val) / total
    test_frac = len(test) / total
    assert abs(train_frac - 0.70) < 0.02
    assert abs(val_frac - 0.15) < 0.02
    assert abs(test_frac - 0.15) < 0.02


def test_both_classes_represented_in_each_split(splits):
    train, val, test = splits
    for df in (train, val, test):
        assert set(df["Class"].unique()) == {0, 1}


def test_legitimate_subset_extraction_works(splits):
    train, _, _ = splits
    normal = train[train["Class"] == 0]
    assert len(normal) > 0
    assert (normal["Class"] == 0).all()


def test_pipeline_reload_is_deterministic(pipeline):
    raw_sample = pd.DataFrame([{
        "Time": 12345.0, "Amount": 55.5,
        **{f"V{i}": float(i) * 0.01 for i in range(1, 29)},
    }])
    out1 = pipeline.transform(raw_sample)

    reloaded = FraudPreprocessor.load(os.path.join(ARTIFACTS_DIR, "preprocessing_pipeline.pkl"))
    out2 = reloaded.transform(raw_sample)

    pd.testing.assert_frame_equal(out1, out2)
