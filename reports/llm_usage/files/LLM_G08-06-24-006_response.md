Wrapping up a complex machine learning pipeline with a robust automated test suite and an interactive, user-friendly GUI is exactly the kind of polish that elevates a third-year computer science project to a professional standard.

Here are the final two modules for your project.

## Part 1: Automated Test Suite (`tests/test_pipeline.py`)

This test suite utilizes `pytest` and its built-in `tmp_path` fixture to ensure all generated files (normalized datasets, models, metrics) are isolated and automatically cleaned up after the tests run. It verifies the core logical constraints, handling of nulls, data normalization, and the output schemas.

```python
import os
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
# Import the pipeline functions (ensure your PYTHONPATH is set correctly)
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

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
```

## Part 2: Graphical User Interface (`src/qubo_project/gui.py`)

This `streamlit` application orchestrates the entire pipeline. Because our backend functions require file paths, the GUI handles uploading files by caching them to a local `outputs/gui_temp/` directory. It uses `st.session_state` to prevent out-of-order execution.

```python
import streamlit as st
import pandas as pd
import json
import os
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

# Configuration and Paths
st.set_page_config(page_title="QUBO Feature Selection Pipeline", layout="wide")
OUTPUT_DIR = "outputs/gui_temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# State Management
for key in ['dataset_path', 'target_col', 'preproc_done', 'fs_done', 'train_done', 'pred_done']:
    if key not in st.session_state:
        st.session_state[key] = False if 'done' in key else None

st.title("⚙️ QUBO Binary Classification Pipeline")
st.markdown("Execute the stages of the machine learning pipeline sequentially.")

# --- Step 1: Data Upload ---
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df_preview = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df_preview.head())
    
    target_col = st.selectbox("Select the Target Column:", df_preview.columns)
    
    if st.button("Confirm Dataset & Target"):
        dataset_path = os.path.join(OUTPUT_DIR, "input_dataset.csv")
        df_preview.to_csv(dataset_path, index=False)
        st.session_state.dataset_path = dataset_path
        st.session_state.target_col = target_col
        st.success(f"Dataset saved. Target set to: `{target_col}`")

st.divider()

# --- Step 2: Preprocessing ---
st.header("2. Preprocessing")
if not st.session_state.dataset_path:
    st.info("Please complete Step 1 first.")
else:
    min_perc_valid = st.slider("Minimum % of valid data required to keep a column:", 0.0, 1.0, 0.05, 0.01)
    
    if st.button("Run Preprocessing"):
        with st.spinner("Normalizing data and dropping sparse columns..."):
            norm_path = os.path.join(OUTPUT_DIR, "normalized.csv")
            stats_path = os.path.join(OUTPUT_DIR, "preproc_stats.json")
            
            fit_normalize(
                input_csv=st.session_state.dataset_path,
                target_column=st.session_state.target_col,
                normalized_csv=norm_path,
                outInitalRes_json=stats_path,
                minPercValid=min_perc_valid
            )
            st.session_state.preproc_done = True
            st.success("Preprocessing Complete!")
            
            with open(stats_path, 'r') as f:
                st.json(json.load(f))

st.divider()

# --- Step 3: QUBO Feature Selection ---
st.header("3. QUBO Feature Selection")
if not st.session_state.preproc_done:
    st.info("Please complete Step 2 first.")
else:
    col1, col2, col3 = st.columns(3)
    perc_selected = col1.number_input("Percentage to Select", 0.01, 1.0, 0.20, 0.01)
    allowance = col2.number_input("Allowance (± features)", 0, 10, 1)
    alpha_comps = col3.number_input("Alpha Computations", 10, 500, 100)
    
    if st.button("Run Feature Selection"):
        with st.spinner("Running Simulated Annealing for QUBO Optimization..."):
            select_features(
                normalized_csv=os.path.join(OUTPUT_DIR, "normalized.csv"),
                reducedTrain_csv=os.path.join(OUTPUT_DIR, "train_reduced.csv"),
                reducedTest_csv=os.path.join(OUTPUT_DIR, "test_reduced.csv"),
                output_ottim_csv=os.path.join(OUTPUT_DIR, "optimizations.csv"),
                output_json=os.path.join(OUTPUT_DIR, "fs_stats.json"),
                target_column=st.session_state.target_col,
                percSelected=perc_selected,
                allowance=allowance,
                alpha_computations=alpha_comps
            )
            st.session_state.fs_done = True
            st.success("Feature Selection Complete!")
            
            with open(os.path.join(OUTPUT_DIR, "fs_stats.json"), 'r') as f:
                stats = json.load(f)
                st.write(f"**Selected {stats['n_selected']} features:**")
                st.write(", ".join(stats['selected_feature_names']))

st.divider()

# --- Step 4: Model Training ---
st.header("4. Model Training")
if not st.session_state.fs_done:
    st.info("Please complete Step 3 first.")
else:
    clf_choice = st.selectbox("Select Classifier:", ["random_forest", "logistic_regression", "gradient_boosting"])
    
    if st.button("Train Model"):
        with st.spinner(f"Training {clf_choice}..."):
            train(
                classifier=clf_choice,
                reducedTrain_csv=os.path.join(OUTPUT_DIR, "train_reduced.csv"),
                target_column=st.session_state.target_col,
                model_path=os.path.join(OUTPUT_DIR, "model.joblib"),
                metrics_json=os.path.join(OUTPUT_DIR, "train_metrics.json")
            )
            st.session_state.train_done = True
            st.success("Training Complete!")

st.divider()

# --- Step 5: Prediction & Visualization ---
st.header("5. Prediction on Test Set")
if not st.session_state.train_done:
    st.info("Please complete Step 4 first.")
else:
    if st.button("Run Predictions"):
        with st.spinner("Evaluating model on test data..."):
            predict(
                reduced_Test_csv=os.path.join(OUTPUT_DIR, "test_reduced.csv"),
                target_column=st.session_state.target_col,
                model_path=os.path.join(OUTPUT_DIR, "model.joblib"),
                predictions_csv=os.path.join(OUTPUT_DIR, "predictions.csv"),
                classif_stats_json=os.path.join(OUTPUT_DIR, "classif_stats.json")
            )
            st.session_state.pred_done = True
            st.success("Predictions Generated!")
            
            # Visualization
            with open(os.path.join(OUTPUT_DIR, "classif_stats.json"), 'r') as f:
                stats = json.load(f)
                
            st.subheader("Classification Results")
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric("Accuracy", f"{stats['accuracy']:.4f}")
            col_met2.metric("ROC-AUC", f"{stats.get('roc_auc', 'N/A')}")
            col_met3.metric("F1 Score (Target=1)", f"{stats['class_1']['f1']:.4f}")
            
            st.write("### Full Statistics Details")
            st.json(stats)