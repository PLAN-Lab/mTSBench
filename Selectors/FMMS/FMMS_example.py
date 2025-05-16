from gen_meta_features import generate_meta_features
import numpy as np 
from joblib import load
import os 
from datasets import load_dataset
import time 
import sys
import pandas as pd 
from FMMS import FMMS
import torch
import config
import utils



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FMMS_example.py <csv_path>")
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

    trained_model_location = '../MetaOD/trained_models'
    meta_scalar = load(os.path.join(trained_model_location,"meta_scalar.joblib"))
    # generate meta features         
    meta_X, _ = generate_meta_features(X)
    # Check and adjust length of meta_vec to 200
    if len(meta_X) < 200:
        # Pad with zeros if meta_vec is too short
        meta_X.extend([0] * (200 - len(meta_X)))
    meta_X = np.nan_to_num(meta_X,nan=0.0, posinf=0.0, neginf=0.0)
    meta_X = meta_scalar.transform(np.asarray(meta_X).reshape(1, -1)).astype(float)
    

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

    ypred = fmms(meta_X).detach().numpy()
    fmms_df = pd.read_csv("FMMS_ypred_0_ts_test.csv")
    # Identify columns containing model scores (excluding the "Data" column)
    model_columns = fmms_df.columns.drop("Data")
    top_3 = ypred[model_columns].sort_values(ascending=False).index[:3].tolist()
    print("top 3 models:")
    print(top_3)

    