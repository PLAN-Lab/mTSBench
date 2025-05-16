#!/bin/bash
# Capture the start time
start_time=$(date +%s)


source env_metaOD/bin/activate
python train_metaod.py > train_meta.txt

# Calculate and print the total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: ${runtime} seconds" >> train_meta_runtime.txt
