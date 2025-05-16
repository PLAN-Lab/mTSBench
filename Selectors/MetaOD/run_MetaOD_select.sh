#!/bin/bash
start_time=$(date +%s)
source env_metaOD/bin/activate

python model_selection.py > selection.txt
# Calculate and print the total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total evalaution time: ${runtime} seconds" >> eval_runtime.txt
