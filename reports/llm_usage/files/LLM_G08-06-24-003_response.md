Here is the code for the `src/qubo_project/model.py` module, satisfying all your technical constraints and the specifications from the project document.

### Implementation Details

* **Classifiers:** The module implements Random Forest (mandatory), alongside Logistic Regression and Gradient Boosting as suggested.
* **Training (`train`):** Reads the training set, times both the I/O and training operations, and exports the model via joblib along with the metrics JSON.
* **Prediction (`predict`):** Evaluates the saved model on the test set, outputting both the granular CSV of predictions (row_n, target, prediction, score) and the comprehensive JSON statistics file containing class-wise metrics, ROC-AUC, and the Confusion Matrix.
* **CLI Architecture:** Uses `argparse.ArgumentParser` with `add_subparsers` to elegantly separate the `train` and `predict` command-line modes.

### Source Code: `model.py`
```python
import argparse
import json
import os
import time
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix
)

def train(
    classifier: str, 
    reducedTrain_csv: str, 
    target_column: str, 
    model_path: str, 
    metrics_json: str, 
    seed: int = 42):
    """
    Trains a binary classifier on the reduced dataset and saves the model.
    """
    # 1. Map string to classifier instance
    classifiers = {
        "random_forest": RandomForestClassifier(random_state=seed),
        "logistic_regression": LogisticRegression(random_state=seed, max_iter=1000),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed)
    }
    
    if classifier not in classifiers:
        raise ValueError(f"Classifier '{classifier}' not supported. Choose from: {list(classifiers.keys())}")
        
    clf = classifiers[classifier]

    # 2. Data Loading
    start_input = time.time()
    df = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.time() - start_input

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. Model Training
    start_train = time.time()
    clf.fit(X, y)
    training_time = time.time() - start_train

    # 4. Save Outputs
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(metrics_json) or '.', exist_ok=True)
    
    joblib.dump(clf, model_path)

    n_samples = len(df)
    target_1_percentage = float((y == 1).mean() * 100)

    stats = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": os.path.basename(reducedTrain_csv),
        "target_column": target_column,
        "model_path": os.path.basename(model_path),
        "n_samples": n_samples,
        "n_features": len(X.columns),
        "target_1_percentage": round(target_1_percentage, 2),
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2)
    }

    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

def predict(
    reduced_Test_csv: str, 
    target_column: str, 
    model_path: str, 
    predictions_csv: str, 
    classif_stats_json: str):
    """
    Loads a trained model, generates predictions on test data, and computes metrics.
    """
    # 1. Data and Model Loading
    df = pd.read_csv(reduced_Test_csv)
    X = df.drop(columns=[target_column])
    y_true = df[target_column]

    clf = joblib.load(model_path)

    # 2. Generate Predictions
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1] # Probability of positive class (1)

    # 3. Create Predictions CSV
    predictions_df = pd.DataFrame({
        "row_n": df.index,
        "target": y_true,
        "prediction": y_pred,
        "score": np.round(y_prob, 2) if 'np' in globals() else [round(p, 2) for p in y_prob]
    })
    
    os.makedirs(os.path.dirname(predictions_csv) or '.', exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = None # Handle edge case where test set might only have one class
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    target_1_count = int((y_true == 1).sum())

    # 5. Save Classification Stats
    stats = {
        "classifier": type(clf).__name__,
        "n_samples": len(df),
        "target_1_count": target_1_count,
        "target_1_percentage": round((target_1_count / len(df)) * 100, 2),
        "accuracy": acc,
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        },
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }

    os.makedirs(os.path.dirname(classif_stats_json) or '.', exist_ok=True)
    with open(classif_stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training and prediction module.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run: 'train' or 'predict'")

    # Subparser for the 'train' command
    parser_train = subparsers.add_parser("train", help="Train a classification model")
    parser_train.add_argument("--classifier", required=True, choices=["random_forest", "logistic_regression", "gradient_boosting"], help="Algorithm to use")
    parser_train.add_argument("--in-reduced", required=True, help="Input reduced training CSV")
    parser_train.add_argument("--target", required=True, help="Target column name")
    parser_train.add_argument("--out-model", required=True, help="Path to save trained model (.joblib)")
    parser_train.add_argument("--out-metrics", required=True, help="Output metrics JSON")
    parser_train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Subparser for the 'predict' command
    parser_predict = subparsers.add_parser("predict", help="Predict using a trained model")
    parser_predict.add_argument("--input-testset", required=True, help="Input reduced test CSV")
    parser_predict.add_argument("--target", required=True, help="Target column name")
    parser_predict.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser_predict.add_argument("--out-predictions", required=True, help="Output predictions CSV")
    parser_predict.add_argument("--out-stats", required=True, help="Output classification stats JSON")

    args = parser.parse_args()

    # Route execution based on subcommand
    if args.command == "train":
        train(
            classifier=args.classifier,
            reducedTrain_csv=args.in_reduced,
            target_column=args.target,
            model_path=args.out_model,
            metrics_json=args.out_metrics,
            seed=args.seed
        )
    elif args.command == "predict":
        predict(
            reduced_Test_csv=args.input_testset,
            target_column=args.target,
            model_path=args.model,
            predictions_csv=args.out_predictions,
            classif_stats_json=args.out_stats
        )
```