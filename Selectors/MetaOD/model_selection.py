import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import pandas as pd
import os
from metaod.models.utility import prepare_trained_model
from metaod.models.predict_metaod import select_model
from scipy.stats import spearmanr
from utils import ndcg

# Load the CSV file
df = pd.read_csv(os.path.join("../results", "performance_matrix_vus_pr_ranked.csv"))

# Function to get ranks for a specific dataset
def get_ranks_for_dataset(dataset_name):
    row = df[df['dataset'] == dataset_name]
    if row.empty:
        print(f"No ranking found for dataset: {dataset_name}")
        return None
    ranks = row.iloc[0, 1:].tolist()
    return ranks

# Function to save ground truth, prediction, ranking correlation, and runtime
def save_ground_truth_prediction_correlation(dataset_name, ground_truth, prediction, rho, runtime, output_file):
    df = pd.DataFrame({
        'dataset': [dataset_name],
        'ground_truth': [ground_truth],
        'prediction': [prediction],
        'spearman_rho': [rho],
        'runtime_seconds': [runtime]
    })
    df.to_csv(output_file, mode='a', index=False, header=not pd.io.common.file_exists(output_file))

if __name__ == "__main__":
    roc_df = pd.read_csv(os.path.join("../results", "performance_matrix_vus_pr.csv"))
    roc_mat = roc_df.to_numpy()
    data_headers = roc_mat[0:, 0]
    print("data_headers")
    print(data_headers)
    print(len(data_headers))
    csv_file_list = data_headers

    for j in range(len(csv_file_list)):
        start_time = time.time()  # Start timing the run

        csv_file = csv_file_list[j]
        print(csv_file)
        csv_file = csv_file.split('.')[0]
        X = pd.read_csv(os.path.join("../Datasets", 'mTSBench', csv_file.split('_')[0],csv_file + '_test.csv'))
        X = X.iloc[:, 1:-1] # excluding timestamp,and label

        meta_scalar = MinMaxScaler()
        X_transformed = meta_scalar.fit_transform(X)
        
        selected_models = select_model(X_transformed, trained_model_location='trained_models/', n_selection=100)
        ground_truth = get_ranks_for_dataset(csv_file)

        if ground_truth is None:
            print(f"Skipping dataset {csv_file} due to missing ground truth.")
            continue

        ground_truth_ranks = {item: rank for rank, item in enumerate(ground_truth, start=1)}
        predicted_ranks = {item: rank for rank, item in enumerate(selected_models, start=1)}

        ground_truth_ranking = [ground_truth_ranks.get(item, len(ground_truth) + 1) for item in ground_truth]
        predicted_ranking = [predicted_ranks.get(item, len(selected_models) + 1) for item in ground_truth]

        rho, _ = spearmanr(ground_truth_ranking, predicted_ranking)

        runtime = time.time() - start_time  # Calculate runtime
        save_ground_truth_prediction_correlation(csv_file, ground_truth, selected_models, rho, runtime, "ground_truth_and_predictions.csv")
        print(f"Spearman's Rank Correlation: {rho}, Runtime: {runtime:.2f} seconds")


