import time
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from metaod.models.predict_metaod import select_model
from datasets import load_dataset

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metaod_example.py <csv_path>")
        sys.exit(1)

    data_name = sys.argv[1]  # e.g., "MSL_C-1"
    start_time = time.time()


        # Load test CSV
    try:
        calit2 = load_dataset("PLAN-Lab/mTSBench", data_dir=data_name)
        # Convert to pandas
        df = calit2["test"].to_pandas()
    except:
        ## Alternatively, if the above does not work, using exact path
        url = "https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/{}/{}_test.csv".format(data_name.split('_')[0], data_name)
        df = pd.read_csv(url)

    X = df.iloc[:, 1:-1]  # remove timestamp and label

    # Normalize
    X_scaled = MinMaxScaler().fit_transform(X)

    # Predict ranking
    selected_models = select_model(X_scaled, trained_model_location='trained_models/', n_selection=100)

    pred_ranks = {item: i+1 for i, item in enumerate(selected_models)}

    print("Top 3 recommended models are {}".format(pred_ranks[:3]))
