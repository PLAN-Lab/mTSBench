#!/bin/bash
start_time=$(date +%s)
source env_metaOD/bin/activate

python  generate_meta_features_FMM.py > feature_train.txt
python generate_meta_features_FMM_test.py > feature_test.txt
# Calculate and print the total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total features time: ${runtime} seconds" >> features_runtime.txt
