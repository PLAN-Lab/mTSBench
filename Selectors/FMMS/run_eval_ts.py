import pandas as pd
import config
import utils
from FMMS import FMMS
import torch
import torch.utils.data as Data
from utils import MinMaxNorm
import numpy as np 
import os 

def main(txt=''):
    rate, data_name = config.get_rate()
    print("########### data_name {} #############".format(data_name))
    df = pd.read_csv("meta_features_FMMS_test_with_runtime.csv")
    
    # Save the 'Data' column to use as row names later
    row_names = df['Dataset']
    
    # Drop the 'Data' column and normalize the data
    df = df.drop(['Dataset'], axis=1)
    df = df.drop(['Runtime'], axis=1)
    df = df.fillna(0)
    df = MinMaxNorm(df)  # Normalization
    df = df.fillna(0)
    X = df.values.astype(np.float32)

    # FMMS model parameters
    optlsit = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adagrad': torch.optim.Adagrad}
    losslist = {'rmse': utils.rmse_loss, 'mse': utils.mse_loss,
                'cos': utils.cos_loss, 'L1': utils.l1_loss,
                'sL1': utils.SmoothL1_loss, 'kd': utils.KLDiv_loss
                }
    params = {
        'embedding_size': 4,
        'feature_size': X.shape[1],
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
        'opt': optlsit[opt],
        'optname': opt,
        'loss': losslist[l],
        'lossname': l,
    }
    path = config.get_para(train_params, params)
    print(path)
    fmms = FMMS(**params)
    fmms.load_state_dict(torch.load("models/FMMS%s_%s.pt" % (path, txt)))
    ypred = fmms(torch.tensor(X))
    ypred = ypred.detach().numpy()
    print("ypred.shape ", ypred.shape)

    # Read just the header row (no data)
    csv_path  = os.path.join("../results_val", "performance_matrix_vus_pr.csv")
    df = pd.read_csv(csv_path, nrows=0)

    # .columns is an Index of column names
    print(list(df.columns))

    # column_names = [f"Prediction_{i}" for i in range(ypred.shape[1])]
    column_names = list(df.columns)[1:] # exclude row name
    # Create the DataFrame
    ypred_df = pd.DataFrame(ypred, columns=column_names)
    
    # Add the row names back as an index column
    ypred_df['Data'] = row_names
    ypred_df.set_index('Data', inplace=True)

    # Save the DataFrame
    output_path = f'FMMS_ypred_{txt}_{data_name}_test.csv'
    ypred_df.to_csv(output_path)
    print(f"ypred saved to '{output_path}'")

if __name__ == '__main__':
    for i in range(1):
        main(str(i))
