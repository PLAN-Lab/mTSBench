# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:13:27 2020

@author: yuezh
"""

import os
import random
import pandas as pd
import numpy as np

from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler

from scipy.io import loadmat

from joblib import dump, load

from metaod.models.utility import read_arff, fix_nan
from metaod.models.gen_meta_features import generate_meta_features
from metaod.models.core import MetaODClass

# read in performance table
roc_df = pd.read_csv(os.path.join("../results_val", 'performance_matrix_vus_pr.csv'))
print(roc_df[:1])
print(roc_df.shape)
# trim the table
roc_mat = roc_df.to_numpy()
roc_mat_red = fix_nan(roc_mat[1:, 1:].astype('float'))
print("roc_mat_red ", roc_mat_red.shape)

# get statistics of the training data
n_datasets, n_configs = roc_mat_red.shape[0], roc_mat_red.shape[1]
print("n_datasets, n_configs ", n_datasets, n_configs)
data_headers = roc_mat[:, 0]
print("data_headers ") #Xiaona: There are only 93 datasets, it was suppose to be 124, check what happended, 
                        #some did not make it because there were no anomalies in the valid or test set after split
print(data_headers)
print("len(data_headers) ", len(data_headers))
config_headers = roc_df.columns[1:]
print("config_headers ", config_headers)
dump(config_headers, 'trained_models/model_list.joblib')

# %%

# build meta-features
meta_csv = np.zeros([len(data_headers), 200])

# read in csv files
csv_file_list = data_headers

# Xiaona: to generate meta-features, the data must have at least three columns, so changed the code
for j in range(len(csv_file_list)):
    csv_file = csv_file_list[j]
    csv_file = csv_file.split('.')[0] #Xiaona: enable more data to be used for meta-features
                                      # the detector are tesed on validation set. 
                                      # we include the train data for a better capturing of the features
    X = pd.read_csv(os.path.join("../Datasets", 'mTSBench', csv_file.split('_')[0],csv_file + '_train.csv'))
    X = X.iloc[:, 1:-1]
    print(X.shape)

    #Xiaona: try transform X to range[0,1]
    meta_scalar = MinMaxScaler()
    X_transformed = meta_scalar.fit_transform(X)


    
    try:
        # Attempt to generate meta features
        # Generate meta-features
        meta_vec, meta_vec_names = generate_meta_features(X_transformed)
        
        # Check and adjust length of meta_vec to 200
        if len(meta_vec) < 200:
            # Pad with zeros if meta_vec is too short
            meta_vec.extend([0] * (200 - len(meta_vec)))
        elif len(meta_vec) > 200:
            # Truncate if meta_vec is too long
            meta_vec = meta_vec[:200]
        
        # Assign to meta_csv array
        meta_csv[j, :] = meta_vec
        print(j, csv_file)
    except Exception as e:
        # Print the error and continue with the next file
        print(f"Error processing {csv_file} at index {j}: {e}")



# use cleaned and transformed meta-features
meta_scalar = MinMaxScaler()
meta_csv_transformed = meta_scalar.fit_transform(meta_csv)
meta_csv_transformed = fix_nan(meta_csv_transformed)
print(meta_csv_transformed.shape)
print(meta_csv_transformed[:5])
print("config_headers ", config_headers)
print("len(config_headers) ", len(config_headers))
print("csv_file_list ", csv_file_list)
print("len(csv_file_list) ", len(csv_file_list))
# Convert meta_csv_transformed to a DataFrame
columns = [f'fea_{i+1}' for i in range(200)]  # Generate column names fea_1, fea_2, ..., fea_200
meta_df = pd.DataFrame(meta_csv_transformed, index=csv_file_list, columns=columns)

# Set 'dataset' as the index
meta_df.set_index('dataset', inplace=True)

# Transpose the DataFrame
df_transposed = meta_df.transpose()

# Reset index to turn the index into a column
df_transposed.reset_index(inplace=True)

# Rename the 'index' column to 'Model'
df_transposed.rename(columns={'index': 'Model'}, inplace=True)
# Save DataFrame to a CSV file
output_csv_path = '../FMMS/meta_features_FMMS.csv'
df_transposed.to_csv(output_csv_path, index_label='Dataset')  # Include row names (csv_file_list) as the "Dataset" column

print(f"Meta-features saved to {output_csv_path}")
