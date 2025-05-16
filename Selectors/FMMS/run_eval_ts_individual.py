import pandas as pd
import config
import evaluation
import utils
from FMMS import FMMS
import torch
import pickle
from utils import MinMaxNorm
import numpy as np
import time
import os

def main(txt=''):
    rate, data_name = config.get_rate()
    print(f"########### Data Name: {data_name} #############")
    df = pd.read_csv("meta_features_FMMS_test_with_runtime.csv")    
    # Save the 'Data' column to use as row names later
    row_names = df['Dataset'].values.tolist() 
    # Drop the 'Data' column and normalize the data
    df = df.drop(['Dataset'], axis=1)
    df = df.drop(['Runtime'], axis=1)
    df = df.fillna(0)
    df = MinMaxNorm(df)  # Normalization
    X = df.values.astype(np.float32)
    # FMMS model parameters
    params = {
        'embedding_size': 4,
        'feature_size':  X.shape[1],
        'model_size': 24,
        'FM': True,
        'DNN': False,
        'layer_size': 3,
        'hiddensize': 64
    }
    opt = 'adam'
    l = 'cos'
    train_params = {
        'batch': 8,
        'lr': 0.005,
        'epoch': 50,
        'opt': {'adam': torch.optim.Adam}[opt],
        'optname': opt,
        'loss': {'cos': utils.cos_loss}[l],
        'lossname': l,
    }
    path = config.get_para(train_params, params)
    print(f"Model Path: {path}")

    # Load FMMS model
    fmms = FMMS(**params)
    fmms.load_state_dict(torch.load("models/FMMS%s_%s.pt" % (path, txt)))

    # Read just the header row (no data)
    csv_path  = os.path.join("../results_val", "performance_matrix_vus_pr.csv")
    df_column = pd.read_csv(csv_path, nrows=0)

    # .columns is an Index of column names
    print(list(df_column.columns))

    # column_names = [f"Prediction_{i}" for i in range(ypred.shape[1])]
    column_names = list(df_column.columns)[1:] # exclude row name
    # Create an empty DataFrame to store results
    all_results = []

    # Process each row individually
    print("X.shape ", df.shape)
    for idx, row in df.iterrows():
        # Extract the row's data and normalize it
        row_data = row.fillna(0).values.astype(np.float32)
        row_tensor = torch.tensor(row_data).unsqueeze(0)  # Add batch dimension

        # Measure runtime
        start_time = time.time()
        ypred = fmms(row_tensor).detach().numpy()
        elapsed_time = time.time() - start_time
        print(f"Processed {row_names[idx]} in {elapsed_time:.4f} seconds")

        # Save prediction and runtime
        result = dict(zip(column_names, ypred[0]))
        result['Data'] = row_names[idx]
        result['Runtime'] = elapsed_time
        all_results.append(result)

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.set_index('Data', inplace=True)
    # Save all results to a single CSV file
    # Save the DataFrame
    output_path = f'FMMS_ypred_{txt}_{data_name}_test_individual.csv'
    results_df.to_csv(output_path)
    print(f"Results saved to '{output_path}'")


if __name__ == '__main__':
    for i in range(1):
        main(str(i))

