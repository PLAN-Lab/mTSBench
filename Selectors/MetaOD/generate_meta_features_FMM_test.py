import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from metaod.models.utility import fix_nan
from metaod.models.gen_meta_features import generate_meta_features
import time

# ########## Generate meta-features on test set #########
# Read performance table
roc_df = pd.read_csv(os.path.join("../results", 'performance_matrix_vus_pr.csv'))
roc_mat = roc_df.to_numpy()
roc_mat_red = fix_nan(roc_mat[1:, 1:].astype('float'))

# Dataset and configuration information
data_headers = roc_mat[:, 0]
meta_csv = np.zeros([len(data_headers), 200])

# CSV file list
csv_file_list = data_headers

# Initialize list to store dataset names and runtimes
runtime_records = []

# Process each dataset
for j, csv_file in enumerate(csv_file_list):
    csv_file = csv_file.split('.')[0]  # Extract base name
    try:
        # Load and preprocess the dataset
        start_time = time.time()
        X = pd.read_csv(os.path.join("../Datasets", 'mTSBench', csv_file.split('_')[0],csv_file + '_test.csv'))
        X = X.iloc[:, 1:-1]  # Remove irrelevant columns
        meta_scalar = MinMaxScaler()
        X_transformed = meta_scalar.fit_transform(X)

        # Generate meta-features
        meta_vec, meta_vec_names = generate_meta_features(X_transformed)

        # Adjust length of meta_vec to 200
        if len(meta_vec) < 200:
            meta_vec.extend([0] * (200 - len(meta_vec)))
        elif len(meta_vec) > 200:
            meta_vec = meta_vec[:200]

        # Assign to meta_csv array
        meta_csv[j, :] = meta_vec

        # Measure runtime
        elapsed_time = time.time() - start_time
        runtime_records.append(elapsed_time)

        print(f"Processed {csv_file} in {elapsed_time:.4f} seconds")

    except Exception as e:
        print(f"Error processing {csv_file} at index {j}: {e}")
        runtime_records.append(None)  # Use None for rows with errors

# Convert meta_csv to a DataFrame
columns = [f'fea_{i+1}' for i in range(200)]  # Generate column names fea_1, fea_2, ..., fea_200
meta_df = pd.DataFrame(meta_csv, index=csv_file_list, columns=columns)

# Add runtime column to the meta-features DataFrame
meta_df['Runtime'] = runtime_records

# Save DataFrame to a CSV file
output_csv_path = '../FMMS/meta_features_FMMS_test_with_runtime.csv'
meta_df.to_csv(output_csv_path, index_label='Dataset')

print(f"Meta-features with runtime saved to {output_csv_path}")

