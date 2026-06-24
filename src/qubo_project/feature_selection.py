import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import dimod
import neal

def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    allowance: int = 1,
    seed: int = 42,
    percSelected: float = 0.20,
    alpha_computations: int = 100
):
    """
    Formulates and solves the Feature Selection QUBO problem using Simulated Annealing,
    dynamically finding the optimal alpha to meet the desired feature percentage.
    """
    # 1. Initialization and Data Loading
    df = pd.read_csv(normalized_csv)
    
    features = [col for col in df.columns if col != target_column]
    n_features = len(features)
    total_samples = len(df)
    target_k = int(round(percSelected * n_features))
    
    # 2. Compute Spearman Correlations
    # We use Spearman correlation as requested, avoiding Pearson's linearity assumption.
    start_q_time = time.time()
    corr_matrix = df.corr(method='spearman').abs()
    corr_target = corr_matrix[target_column].drop(index=target_column)
    corr_features = corr_matrix.drop(index=target_column, columns=target_column)
    q_matrix_creation_time = time.time() - start_q_time

    # 3. QUBO Optimization Setup
    sampler = neal.SimulatedAnnealingSampler()
    
    # Binary search bounds for alpha
    low_alpha, high_alpha = 0.0, 1.0
    best_alpha = 0.5
    best_selected_vector = []
    best_n_selected = -1
    
    optimization_logs = []
    optimization_times = []

    # 4. Optimization Loop (Varying Alpha)
    for iteration in range(alpha_computations):
        alpha = (low_alpha + high_alpha) / 2.0
        
        # Build the QUBO matrix Q
        Q = {}
        for i in range(n_features):
            feat_i = features[i]
            
            # Diagonal terms: -alpha * influence + (1-alpha) * self-dependence
            # Since self-correlation is 1, |rho_ii| = 1
            Q[(feat_i, feat_i)] = -alpha * corr_target[feat_i] + (1 - alpha)
            
            # Off-diagonal terms: 2 * (1-alpha) * |rho_ij|
            # Multiplied by 2 because dimod takes upper triangular sum(Q_ij * x_i * x_j)
            for j in range(i + 1, n_features):
                feat_j = features[j]
                Q[(feat_i, feat_j)] = 2 * (1 - alpha) * corr_features.loc[feat_i, feat_j]

        # Solve the QUBO problem
        start_opt_time = time.time()
        response = sampler.sample_qubo(Q, seed=seed)
        opt_time = time.time() - start_opt_time
        optimization_times.append(opt_time)

        # Parse results
        sample = response.first.sample
        energy = response.first.energy
        selected_vector = [sample[feat] for feat in features]
        n_selected = sum(selected_vector)
        
        # Log the iteration
        optimization_logs.append({
            "alpha": alpha,
            "optimization_time": opt_time,
            "number_of_features": n_selected,
            "cost_function_value": energy
        })

        # Save state if it's the closest we've gotten to target_k
        if abs(n_selected - target_k) < abs(best_n_selected - target_k) or best_n_selected == -1:
            best_alpha = alpha
            best_selected_vector = selected_vector
            best_n_selected = n_selected

        # Evaluate stopping criteria or adjust search bounds
        if abs(n_selected - target_k) <= allowance:
            break
        elif n_selected < target_k:
            # Selected too few features -> Need more influence -> Increase alpha
            low_alpha = alpha
        else:
            # Selected too many features -> Need more independence -> Decrease alpha
            high_alpha = alpha

    # 5. Process Final Dataset Split
    selected_feature_names = [features[i] for i in range(n_features) if best_selected_vector[i] == 1]
    columns_to_keep = selected_feature_names + [target_column]
    
    # Clean cut at sample M
    M = total_samples - int(round(total_samples * percTest))
    train_df = df.iloc[:M][columns_to_keep]
    test_df = df.iloc[M:][columns_to_keep]
    
    # 6. Save Outputs
    os.makedirs(os.path.dirname(reducedTrain_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_ottim_csv) or '.', exist_ok=True)
    
    train_df.to_csv(reducedTrain_csv, index=False)
    test_df.to_csv(reducedTest_csv, index=False)
    
    # Save optimization iterations
    logs_df = pd.DataFrame(optimization_logs)
    logs_df = logs_df.sort_values(by="alpha", ascending=True)
    logs_df.to_csv(output_ottim_csv, index=False)

    # Save statistics JSON
    stats = {
        "n_features": n_features,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": int(best_n_selected), # <-- Int()
        "alpha": round(best_alpha, 5),
        "selected_vector": [int(x) for x in best_selected_vector], # <-- List comprehension per convertire ogni elemento
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(optimization_logs),
        "percTest": percTest,
        "training_dataset_size": len(train_df),
        "test_dataset_size": len(test_df),
        "q_matrix_creation_time": round(q_matrix_creation_time, 3),
        "mean_optimization_time": round(float(np.mean(optimization_times)), 3),
        "std_dev_optimization_time": round(float(np.std(optimization_times)), 3)
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature reduction via QUBO formulation.")
    
    parser.add_argument("--in-normalized", required=True, help="Input normalized dataset CSV")
    parser.add_argument("--out-train", required=True, help="Output reduced training CSV")
    parser.add_argument("--out-test", required=True, help="Output reduced test CSV")
    parser.add_argument("--out-optimizations", required=True, help="Output optimizations log CSV")
    parser.add_argument("--out-json", required=True, help="Output stats JSON")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--perc-selected", type=float, default=0.20, help="Percentage of features to select")
    parser.add_argument("--allowance", type=int, default=1, help="Tolerance for number of selected features")
    parser.add_argument("--perc-test", type=float, default=0.30, help="Percentage of test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulated annealing")
    parser.add_argument("--alpha-computations", type=int, default=100, help="Max iterations for alpha search")
    
    args = parser.parse_args()
    
    select_features(
        normalized_csv=args.in_normalized,
        reducedTrain_csv=args.out_train,
        reducedTest_csv=args.out_test,
        output_ottim_csv=args.out_optimizations,
        output_json=args.out_json,
        target_column=args.target,
        percTest=args.perc_test,
        allowance=args.allowance,
        seed=args.seed,
        percSelected=args.perc_selected,
        alpha_computations=args.alpha_computations
    )