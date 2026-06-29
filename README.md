# 🚀 QUBO-Optimized Binary Classification Pipeline

**Software Engineering Project - A.Y. 2025/2026**

**Group:** G08

**Members:** 60/61/66480, 60/61/66463

This repository hosts a complete Python-based Machine Learning pipeline developed for binary classification tasks (e.g., credit risk assessment). The distinctive feature of this software is the integration of a **Quadratic Unconstrained Binary Optimization (QUBO)** module to perform advanced dimensionality reduction.

---

## 🧠 Pipeline Workflow

The system is designed to execute a rigid, reproducible workflow:

1. **Data Cleaning & Z-Score Normalization:** Parses the input dataset, removes highly sparse features (based on a user-defined threshold), and standardizes the numeric distribution of the variables.
2. **QUBO Feature Selection:** Solves a QUBO formulation via Simulated Annealing to isolate the most relevant predictors. The cost function dynamically balances the Spearman correlation (feature-target influence) against collinearity (inter-feature independence) tuning the alpha penalty parameter.
3. **Classifier Training:** Ingests the reduced features to train a robust classification model (Random Forest is supported by default).
4. **Inference & Metrics:** Tests the generated model on unseen data, exporting predictions and a comprehensive suite of evaluation metrics (F1-Score, ROC-AUC, Accuracy, and Confusion Matrices).

---

## 🛠️ System Requirements & Setup

* **Environment:** Python 3.11+
* **Dependencies:** Defined in the `requirements.txt` file.

Clone the repository and install the required packages:

    git clone https://github.com/zorenm7/isw-qubo-classification-G08.git
    cd isw-qubo-classification-G08
    pip install -r requirements.txt

---

## 📂 Repository Layout

The project strictly adheres to the requested architectural structure:

    isw-qubo-classification-G08/
    ├── README.md
    ├── requirements.txt
    ├── group_info.yaml
    ├── data/
    │   └── sample_test_dataset.csv         # Mock dataset for automated testing
    ├── src/
    │   └── qubo_project/
    │       ├── __init__.py
    │       ├── preprocessing.py            # Phase 1
    │       ├── feature_selection.py        # Phase 2
    │       ├── model.py                    # Phase 3 & 4
    │       └── gui.py                      # Streamlit GUI
    ├── tests/
    │   └── test_pipeline.py                # Pytest suite
    ├── outputs/                            
    ├── reports/
    │   └── project_report.yaml             # Final documentation
    └── llm_usage/
        └── (Markdown logs of LLM interactions)

---

## 🖥️ Streamlit Dashboard (GUI)

For a user-friendly experience, the entire pipeline can be orchestrated through an interactive web interface built with Streamlit. 

To launch the dashboard from the repository root:

    streamlit run src/qubo_project/gui.py

---

## 🧪 Automated Test Suite

A comprehensive test suite is provided to automatically validate data integrity, QUBO vector generation, and execution constraints. Tests are executed against the `sample_test_dataset.csv` mock file.

To run the validation suite:

    pytest tests/

---

## 💻 CLI Execution Instructions

For automated evaluation and reproducibility, all modules expose a Command Line Interface (CLI). The pipeline must be executed sequentially from the root directory.

### Step 1: Preprocessing

    python src/qubo_project/preprocessing.py \
        --input data/input_dataset.csv \
        --target target \
        --out-data outputs/normalized.csv \
        --out-json outputs/preprocessing_result.json \
        --min-perc-valid 0.05

### Step 2: Feature Selection (QUBO)

    python src/qubo_project/feature_selection.py \
        --in-normalized outputs/normalized.csv \
        --out-train outputs/training_reduced.csv \
        --out-test outputs/test_reduced.csv \
        --out-optimizations outputs/optimizations.csv \
        --out-json outputs/feature_selection_result.json \
        --target target \
        --perc-selected 0.20 \
        --allowance 1 \
        --perc-test 0.30 \
        --seed 42 \
        --alpha-computations 100

### Step 3: Training

    python src/qubo_project/model.py train \
        --classifier random_forest \
        --in-reduced outputs/training_reduced.csv \
        --target target \
        --out-model outputs/model.joblib \
        --out-metrics outputs/training_metrics.json \
        --seed 42

### Step 4: Inference

    python src/qubo_project/model.py predict \
        --input-testset outputs/test_reduced.csv \
        --target target \
        --model outputs/model.joblib \
        --out-predictions outputs/predictions.csv \
        --out-stats outputs/classification_stats.json
