#' Fit a uregression model recommender
#'
#'Model recommender training. This model approximates the performance matrix using SVD and uses Random Forest to train regression models to get U from metafeatures.
#' @param train_performance A data frame or matrix corresponding one row per dataset and one column per configuration, with performance values on each cell.
#' @param train_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#' @param n_factors Number of factors to use for the SVD decomposition.
#' @param ... Extra parameters to pass to the Random Forest model.
#'
#' @return A list containing three elements: 1.`models`, one Random Forest per `n_factors`, 2. `dv`, the matrix multiplication of `DV'`,
#' used internally for prediction and 3. `configurations`, the names of the models, used for prediction.
#' @export
#'
#' @examples
#'
#' dat <- get_data()
#' ureg <- recommender('u_regression_rf')
#' ureg <- ureg %>% fit(dat$train$performance, dat$train$metafeatures, 5)
#' ureg %>% predict(dat$test$metafeatures)
fit.u_regression <- function(object,
                             train_performance,
                             train_metafeatures,
                             n_factors = 5,
                             ...){
  if(ncol(train_performance) < n_factors) n_factors <- ncol(train_performance)
  # Generating the SVD:
  # u will be used for the training
  # dv will be stored for prediction
  d <- svd(x = train_performance,
           nu = n_factors,
           nv = n_factors)
  d$d <- diag(d$d[1:n_factors])
  d$v <- t(d$v)
  dv <- d$d %*% d$v
  # Training the regressors
  models <- purrr::map(tidyr::as_tibble(d$u), function(y, ...){
    ranger::ranger(formula = y ~ .,
                   data = cbind(train_metafeatures, y))
  }, ...)
  res <- list(models = models,
              dv = dv,
              configurations = names(train_performance),
              recommender_type = 'URegression (RF)')
  class(res) <- class(object)
  return(res)
}



#' Recommend models for new datasets using uRegression
#'
#' Model recommendation. Issues a single model recommendation per row in `test_metafeatures`.
#'
#' @param object Object obtained from `uregression_rf_fit()`.
#' @param test_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#'
#' @return A character string corresponding to the recommended model per row on `test_metafeatures`.
#' @export
#'
#' @examples
#'
#' dat <- get_data()
#' ureg <- recommender('u_regression_rf')
#' ureg <- ureg %>% fit(dat$train$performance, dat$train$metafeatures, 5, nrounds = 20)
#' ureg %>% predict(dat$test$metafeatures)
# predict.u_regression <- function(object, test_metafeatures, score = F){
#   predictions <- purrr::map_dfc(object$models, function(x){
#     predict(x, test_metafeatures)$predictions
#   }) %>% as.matrix
#   res <- object$configurations[apply(predictions %*% object$dv, 1, which.max)]
#   if(score){
#     return(dplyr::tibble(score = apply(predictions %*% object$dv, 1, max),
#                          configuration = res))
#   }else{
#     class(res) <- c('recommendation', 'character')
#     return(res)
#   }
# }


predict.u_regression <- function(object, test_metafeatures, score = FALSE, k = 5) {
  # Step 1: Collect predictions from all base models
  predictions <- purrr::map_dfc(object$models, function(x) {
    predict(x, test_metafeatures)$predictions
  }) %>% as.matrix()
  
  # Step 2: Compute configuration scores
  final_scores <- predictions %*% object$dv  # [n_samples x n_configurations]
  
  # Step 3: Extract top-k indices and map to configuration names
  top_k_configs <- apply(final_scores, 1, function(row_scores) {
    top_indices <- order(row_scores, decreasing = TRUE)[1:k]
    object$configurations[top_indices]
  })
  
  # With scores
  if (score) {
    top_k_scores <- apply(final_scores, 1, function(row_scores) {
      sort(row_scores, decreasing = TRUE)[1:k]
    })
    
    # Build and print results
    for (i in 1:ncol(top_k_configs)) {
      cat(paste0("\n📄 Test Sample ", i, " — Top ", k, " configs with scores:\n"))
      print("u_regression")
      print(tibble::tibble(
        model = top_k_configs[, i],
        score = top_k_scores[, i]
      ))
    }
    
    # Return all in a wide-format tibble
    top_k_df <- purrr::map2_dfr(
      1:ncol(top_k_configs),
      1:ncol(top_k_scores),
      ~ tibble::tibble(
        rec_1 = top_k_configs[1, .x], score_1 = top_k_scores[1, .x],
        rec_2 = top_k_configs[2, .x], score_2 = top_k_scores[2, .x],
        rec_3 = top_k_configs[3, .x], score_3 = top_k_scores[3, .x],
        rec_4 = top_k_configs[4, .x], score_4 = top_k_scores[4, .x],
        rec_5 = top_k_configs[5, .x], score_5 = top_k_scores[5, .x]
      )
    )
    
    return(top_k_df)
  }
  
  # Without scores
  top_k_df <- as.data.frame(t(top_k_configs))
  colnames(top_k_df) <- paste0("rec_", 1:k)
  
  # Print
  cat("🔍 Top", k, "recommendations per test sample:\n")
  print(top_k_df)
  
  class(top_k_df) <- c("recommendation_list", "data.frame")
  return(top_k_df)
}


