#' @export
analyze_split <- function(dat_split, rec_type){
  # Get the train/test data
  d <- get_data_from_split(dat_split)
  testing <- rsample::testing(dat_split) #Used for the evaluation metadata
  # Process metafeatures if needed
  processed_metafeatures <- process_metafeatures(d$train$metafeatures,
                                                 d$test$metafeatures,
                                                 rec_type)
  d$train$metafeatures <- processed_metafeatures$train
  d$test$metafeatures <- processed_metafeatures$test

  # Apply and evaluate
  res <- fit_and_evaluate_recommender(rec_type,
                                      d$train$performance,
                                      d$train$metafeatures,
                                      d$test$performance,
                                      d$test$metafeatures)
  res  %>%
    mutate(provider = rsample::testing(dat_split)$provider,
           dataset = rsample::testing(dat_split)$dataset,
           recommender = rec_type)
}




