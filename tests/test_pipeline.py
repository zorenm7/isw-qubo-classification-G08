import os
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import the pipeline functions (ensure your PYTHONPATH is set correctly)
from src.qubo_project.preprocessing import fit_normalize
from src.qubo_project.feature_selection import select_features
from src.qubo_project.model import train, predict

# --- Fixtures ---

@pytest.fixture
def mock_dataset_path(tmp_path):
    """
    Ensures a sample dataset exists. If data/sample_test_dataset.csv is missing 
    in the environment, it creates a robust temporary one for testing.
    """
    target_path = Path("data/sample_test_dataset.csv")
    
    if target_path.exists():
        return str(target_path)
    
    # Fallback: create a dummy dataset with some nulls and zeros for testing
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100) * 5,
        "feature_3_nulls": [np.nan] * 98 + [1.0, 2.0], # > 95% missing
        "feature_4_zeros": [0.0] * 96 + [1.0, 2.0, 3.0, 4.0],
        "feature_5": np.random.randn(100),
        "target": np.random.randint(0, 2, 100)
    })
    
    dummy_path = tmp_path / "sample_test_dataset.csv"
    df.to_csv(dummy_path, index=False)
    return str(dummy_path)


# --- Tests ---

def test_preprocessing_numerical_and_nulls(mock_dataset_path, tmp_path):
    normalized_csv = str(tmp_path / "normalized.csv")
    out_json = str(tmp_path / "preproc_stats.json")
    
    fit_normalize(
        input_csv=mock_dataset_path,
        target_column="target",
        normalized_csv=normalized_csv,
        outInitalRes_json=out_json,
        minPercValid=0.05
    )
    
    df_norm = pd.read_csv(normalized_csv)
    
    # 1. Check only numerical columns
    assert all(pd.api.types.is_numeric_dtype(df_norm[col]) for col in df_norm.columns)
    
    # 2. Check null handling (feature_3_nulls should be dropped)
    with open(out_json, 'r') as f:
        stats = json.load(f)
    assert "feature_3_nulls" in stats["dropped_feature_names"]

def test_normalization(mock_dataset_path, tmp_path):
    normalized_csv = str(tmp_path / "normalized.csv")
    out_json = str(tmp_path / "preproc_stats.json")
    
    fit_normalize(mock_dataset_path, "target", normalized_csv, out_json)
    
    df_norm = pd.read_csv(normalized_csv)
    features = [c for c in df_norm.columns if c != "target"]
    
    # 3. Check z-score standard scaling (mean ~ 0, std ~ 1)
    for col in features:
        assert np.isclose(df_norm[col].mean(), 0, atol=1e-7)
        assert np.isclose(df_norm[col].std(ddof=0), 1, atol=1e-2) or np.isclose(df_norm[col].std(ddof=1), 1, atol=1e-2)

def test_feature_selection_vector_and_percentage(mock_dataset_path, tmp_path):
    # Setup: need normalized data first
    norm_csv = str(tmp_path / "normalized.csv")
    fit_normalize(mock_dataset_path, "target", norm_csv, str(tmp_path / "p.json"))
    
    train_csv = str(tmp_path / "train.csv")
    test_csv = str(tmp_path / "test.csv")
    opt_csv = str(tmp_path / "opt.csv")
    stats_json = str(tmp_path / "fs_stats.json")
    
    select_features(
        normalized_csv=norm_csv, reducedTrain_csv=train_csv, reducedTest_csv=test_csv,
        output_ottim_csv=opt_csv, output_json=stats_json, target_column="target",
        percSelected=0.50, allowance=1, alpha_computations=10 # Higher perc for small datasets
    )
    
    with open(stats_json, 'r') as f:
        stats = json.load(f)
        
    # 4. Binary vector check
    vector = stats["selected_vector"]
    assert all(v in [0, 1] for v in vector)
    
    # 5. Selected percentage approx check
    target_k = stats["target_k"]
    n_selected = stats["n_selected"]
    assert abs(n_selected - target_k) <= stats["allowance"]

def test_model_training_and_prediction(mock_dataset_path, tmp_path):
    # Setup entire pipeline up to test
    norm_csv = str(tmp_path / "normalized.csv")
    fit_normalize(mock_dataset_path, "target", norm_csv, str(tmp_path / "p.json"))
    
    train_csv = str(tmp_path / "train.csv")
    test_csv = str(tmp_path / "test.csv")
    select_features(norm_csv, train_csv, test_csv, str(tmp_path / "o.csv"), str(tmp_path / "s.json"), "target", percSelected=0.99, alpha_computations=5)
    
    model_path = str(tmp_path / "model.joblib")
    metrics_json = str(tmp_path / "train_metrics.json")
    
    # Train
    train("random_forest", train_csv, "target", model_path, metrics_json)
    
    # 6. Check model saved
    assert os.path.exists(model_path)
    
    pred_csv = str(tmp_path / "preds.csv")
    stats_json = str(tmp_path / "pred_stats.json")
    
    # Predict
    predict(test_csv, "target", model_path, pred_csv, stats_json)
    
    # 7. Check prediction CSV format
    df_pred = pd.read_csv(pred_csv)
    expected_cols = ["row_n", "target", "prediction", "score"]
    assert list(df_pred.columns) == expected_cols