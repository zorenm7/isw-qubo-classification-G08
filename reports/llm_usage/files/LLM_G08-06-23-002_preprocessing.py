import argparse
import json
import os
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str,
    minPercValid: float = 0.05
):
    """
    Reads a CSV dataset, drops features with too many missing or zero values,
    normalizes the remaining features using z-score, and saves the results.
    """
    # 1. Dataset Input Phase
    start_input_time = time.time()
    
    # Read the dataset
    df = pd.read_csv(input_csv)
    
    end_input_time = time.time()
    dataset_input_time = end_input_time - start_input_time

    # 2. Processing Phase
    start_processing_time = time.time()
    
    dataset_size = len(df)
    
    # Separate features from the target column
    all_features = [col for col in df.columns if col != target_column]
    n_input_features = len(all_features)
    
    # Evaluate valid data (non-missing and non-zero)
    is_valid = df[all_features].notna() & (df[all_features] != 0)
    valid_percentages = is_valid.mean()
    
    # Determine features to keep and drop based on the minPercValid threshold
    features_to_keep = valid_percentages[valid_percentages >= minPercValid].index.tolist()
    dropped_feature_names = list(set(all_features) - set(features_to_keep))
    
    # Apply z-score normalization on the kept features
    if features_to_keep:
        scaler = StandardScaler()
        df[features_to_keep] = scaler.fit_transform(df[features_to_keep])
        
    end_processing_time = time.time()
    dataset_processing_time = end_processing_time - start_processing_time

    # 3. Output Phase
    
    # Ensure output directories exist (no absolute paths required)
    os.makedirs(os.path.dirname(normalized_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(outInitalRes_json) or '.', exist_ok=True)
    
    # Save the normalized dataset (keeping original headers and target column)
    output_columns = features_to_keep + ([target_column] if target_column in df.columns else [])
    df[output_columns].to_csv(normalized_csv, index=False)
    
    # Prepare and save statistics in JSON format
    stats = {
        "n_input_features": n_input_features,           #
        "n_kept_features": len(features_to_keep),       #
        "dataset_size": dataset_size,                   #
        "dataset_input_time": round(dataset_input_time, 3),        #
        "dataset_processing_time": round(dataset_processing_time, 3),  #
        "dropped_feature_names": dropped_feature_names  #
    }
    
    with open(outInitalRes_json, 'w', encoding='utf-8') as json_file:
        json.dump(stats, json_file, indent=4)

if __name__ == "__main__":
    # Command Line Interface
    parser = argparse.ArgumentParser(description="Data preprocessing and normalization module.")
    
    # Define required CLI arguments
    parser.add_argument("--input", required=True, help="Input dataset name")
    parser.add_argument("--target", required=True, help="Column name of target")
    parser.add_argument("--out-data", required=True, help="Name of output normalized data set")
    parser.add_argument("--out-json", required=True, help="Name of output statistics and data file")
    parser.add_argument("--min-perc-valid", type=float, default=0.05, help="Minimum % of valid non-zero data for a column")
    
    args = parser.parse_args()
    
    # Execute the core function
    fit_normalize(
        input_csv=args.input,
        target_column=args.target,
        normalized_csv=args.out_data,
        outInitalRes_json=args.out_json,
        minPercValid=args.min_perc_valid
    )