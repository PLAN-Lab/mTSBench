# -*- coding: utf-8 -*-
"""
This function is adapted from [MetaOD] by [yuezh]
Original source: [https://github.com/yzhao062/MetaOD]
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
data_headers = roc_mat[0:, 0]
print("data_headers ") #Xiaona: There are only 93 datasets, it was suppose to be 124, check what happended, 
                        #some did not make it because there were no anomalies in the valid or test set after split
print(data_headers)
config_headers = roc_df.columns[1:]
print("config_headers ", config_headers)
folder_path = "trained_models"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)
dump(config_headers, 'trained_models/model_list.joblib')

# %%

# build meta-features
meta_csv = np.zeros([n_datasets, 200])

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
dump(meta_scalar, 'trained_models/meta_scalar.joblib')
min_value = np.min(meta_csv_transformed)
max_value = np.max(meta_csv_transformed)

print(f"Minimum value: {min_value}")
print(f"Maximum value: {max_value}")

if min_value >= 0 and max_value <= 1:
    print("All values are within the range [0, 1].")
else:
    print("Some values are outside the range [0, 1].")

# meta_scalar= load('trained_models/meta_scalar_ours.joblib')
# meta_csv_transformed = meta_scalar.fit_transform(meta_csv)
# meta_csv_transformed = fix_nan(meta_csv_transformed)
# %% train model

# split data into train and valid
seed = 0
full_list = list(range(n_datasets))
random.Random(seed).shuffle(full_list)
n_train = int(0.85 * n_datasets)

print("roc_mat_red ", roc_mat_red.shape)
train_index = full_list[:n_train]
valid_index = full_list[n_train:]

# performance matrix 
train_set = roc_mat_red[train_index, :].astype('float64')
valid_set = roc_mat_red[valid_index, :].astype('float64')

# meta features 
train_meta = meta_csv_transformed[train_index, :].astype('float64')
valid_meta = meta_csv_transformed[valid_index, :].astype('float64')

clf = MetaODClass(train_set, valid_performance=valid_set, n_factors=30,
                  learning='sgd')
clf.train(n_iter=50, meta_features=train_meta, valid_meta=valid_meta,
          learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1,
          n_steps=8)

dump(clf, 'trained_models/train_' + str(seed) + '.joblib')
