# Record the start time
start_time <- Sys.time()
library(ADRecommender)
library(readr)
library(dplyr)

# Define the input directory
input_dir <- "../Datasets/mTSBench/"

# List all *_train.csv files in all subdirectories
csv_files <- list.files(input_dir, pattern = "_train\\.csv$", recursive = TRUE, full.names = TRUE)

# Print the result
print(csv_files)

# Placeholder for the combined metafeatures
all_metafeatures <- list()

# Iterate over each file
for (file in csv_files) {
  cat("Processing file:", file, "\n")
  
  tryCatch({
    # Read the file
    test_data <- read_csv(file)
    
    # Exclude the "is_anomaly" column if it exists
    if ("is_anomaly" %in% colnames(test_data)) {
      test_data <- test_data %>% select(-is_anomaly)
    }
    
    # Replace NA and NaN with 0
    test_data <- test_data %>%
      mutate(across(everything(), ~ replace(., is.na(.) | is.nan(.), 0)))
    
    # Identify columns with zero variance
    zero_variance_cols <- sapply(test_data, function(col) length(unique(col)) <= 1)
    
    # Remove these columns
    test_data <- test_data[, !zero_variance_cols]
    
    # If more than 25,000 rows, keep only the middle 25,000
    if (nrow(test_data) > 25000) {
      start_row <- floor((nrow(test_data) - 25000) / 2) + 1
      test_data <- test_data[start_row:(start_row + 24999), ]
    }
    
    # If fewer than 500 rows, duplicate to 1000
    if (nrow(test_data) < 500) {
      reps <- ceiling(1000 / nrow(test_data))
      test_data <- test_data %>% slice(rep(1:n(), reps)) %>% slice(1:1000)
    }
    
    # Extract metafeatures
    meta_features <- get_metafeatures(
      test_data,
      include_metaod = FALSE,
      include_catch22 = TRUE,
      time_var = 'timestamp'
    )
    
    # Add dataset name
    meta_features$Dataset <- basename(file)
    
    # Append to the list
    all_metafeatures[[file]] <- meta_features
  }, error = function(e) {
    cat("❌ Error processing file:", file, "\n")
    cat("   ->", conditionMessage(e), "\n")
  })
}


# Combine all metafeatures into one data frame
combined_metafeatures <- bind_rows(all_metafeatures)

# Save the combined metafeatures to a CSV file
output_file <-"train_meta_features_orthus.csv"
write_csv(combined_metafeatures, file = output_file)

cat("All metafeatures saved to:", output_file, "\n")

# Record the end time
end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
print(runtime)

#> print(runtime)
#[1] 821.219

