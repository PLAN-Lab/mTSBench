# How to run

1. `run_FMMS.sh`: `run_FMMS.py` trains the selector, and `run_eval_ts.py` makes selections on the test set. Note that this method takes meta-features and prediction the performance on the entire set, so we don't really know the individual runtime for each ts. 

## Prepare features.csv and target.csv 

As the original meta-feature extractor from FMMS does not apply to time series data (see [original repo](https://github.com/bettyzry/FMMS) for details), we used meta-features from MetaOD instead. 

1. Using the code in MetaOD to generate meta-features. `run_get_features_for_FMM.sh`
