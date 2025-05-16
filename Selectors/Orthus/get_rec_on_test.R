# ----------------------------------------
# Train Orthus recommender
# ----------------------------------------
start_time <- Sys.time()
library(ADRecommender)
library(readr)
library(dplyr)
library(purrr)

# Load performance & meta‐feature matrices
auc_matrix           <- read_csv("perfor_matrix_val/performance_matrix_vus_pr.csv")
auc_matrix_numeric   <- auc_matrix[, -1]                            # drop “Dataset” column
meta_matrix          <- read_csv("train_meta_features_orthus.csv")
meta_matrix_numeric  <- meta_matrix[, -ncol(meta_matrix)]           # drop last column if it’s an ID/timestamp

# Fit the Orthus model
rec_our <- recommender("orthus") %>%
  fit(auc_matrix_numeric, meta_matrix_numeric)

end_time <- Sys.time()
cat("Training time:",
    round(as.numeric(difftime(end_time, start_time, units = "secs")), 2),
    "seconds\n\n")


# ----------------------------------------
# Loop over test files, extract meta‐features, predict top-3, save results
# ----------------------------------------
input_dir   <- "../Datasets/mTSBench/"
csv_files   <- list.files(input_dir,
                          pattern = "_test\\.csv$",
                          recursive = TRUE,
                          full.names = TRUE)

output_path <- "recommended_models_with_runtime.csv"

# Initialize or load existing results
if (file.exists(output_path)) {
  existing_results <- read_csv(output_path, show_col_types = FALSE)
} else {
  existing_results <- tibble(
    FileName         = character(),
    RecommendedModel = character(),
    RunTimeSeconds   = numeric()
  )
}

for (file in csv_files) {
  file_basename <- basename(file)
  
  # Skip if already processed
  if (file_basename %in% existing_results$FileName) {
    cat("⏭ Skipping:", file_basename, "\n")
    next
  }
  
  tryCatch({
    t0  <- Sys.time()
    
    # 1) Read data, drop last column (label), handle NaNs/Infs, remove constant cols
    df   <- read.csv(file)
    data <- df[, -ncol(df)] %>%
      mutate(across(everything(), ~ ifelse(is.na(.) | is.nan(.) | is.infinite(.), 0, .))) %>%
      select(where(~ n_distinct(.) > 1))
    
    # 2) If too long, take a centered 25 k window
    if (nrow(data) > 25000) {
      start_row <- floor((nrow(data) - 25000) / 2) + 1
      data      <- data[start_row:(start_row + 24999), ]
    }
    
    # 3) Safe feature extraction (fallback to zeros on error)
    metafeatures <- tryCatch({
      get_metafeatures(
        data,
        include_metaod  = FALSE,
        include_catch22 = TRUE,
        time_var        = "timestamp"
      )
    }, error = function(e) {
      cat("⚠️ Feature‐extraction failed on", file_basename, ":", conditionMessage(e), "\n")
      zero_vec <- rep(0, ncol(meta_matrix_numeric))
      names(zero_vec) <- colnames(meta_matrix_numeric)
      as_tibble(as.list(zero_vec))
    })
    
    # 4) Clean up any NA/NaN
    metafeatures <- metafeatures %>%
      mutate(across(everything(), ~ ifelse(is.na(.) | is.nan(.), 0, .)))
    
    # 5) Predict top‐3 or skip if all zeros
    if (all(unlist(metafeatures) == 0)) {
      model <- NA_character_
      cat("⏭ Skipping prediction — no meta-feature signal\n")
    } else {
      raw_model_list <- rec_our %>% predict(metafeatures)
      top3_vec       <- raw_model_list[[1]]            # extract the first (only) element
      model          <- paste(top3_vec, collapse = ";") # combine into “mod1;mod2;mod3”
      cat("✅ Processed:", file_basename, "→ Top-3:", model, "\n")
    }
    
    t1      <- Sys.time()
    runtime <- as.numeric(difftime(t1, t0, units = "secs"))
    
    # 6) Append and save
    new_row <- tibble(
      FileName         = file_basename,
      RecommendedModel = model,
      RunTimeSeconds   = runtime
    )
    existing_results <- bind_rows(existing_results, new_row)
    write_csv(existing_results, output_path)
    
  }, error = function(e) {
    cat("❌ Error on", file_basename, ":", conditionMessage(e), "\n")
  })
}

cat("\nAll done – results in", output_path, "\n")

