#!/bin/bash
start_time=$(date +%s)
source env_FMMS/bin/activate
python run_FMMS.py > example.txt
# Calculate and print the total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Train time : ${runtime} seconds" >> runtime.txt
start_time=$(date +%s)
python run_eval_ts_individual.py > eval.txt
# Calculate and print the total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Eval time : ${runtime} seconds" >> runtime.txt
