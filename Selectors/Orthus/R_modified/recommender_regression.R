#' Fit a regression model recommender
#'
#'Model recommender training. This model trains Random Forest regression models directly from the metafeatures to the performance matrix.
#' @param train_performance A data frame or matrix corresponding one row per dataset and one column per configuration, with performance values on each cell.
#' @param train_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#' @param ... Extra parameters to pass to the Random Forest model.
#'
#' @return A list containing two elements: 1.`models`, one Random Forest per `n_factors`, 2. `configurations`, the names of the models, used for prediction.
#' @export
#'
#' @examples
#'
#' dat <- get_sample_data()
#' reg <- recommender('regression_rf')
#' reg <- reg %>% fit(dat$train$performance, dat$train$metafeatures, nrounds = 20)
#' reg %>% predict(dat$test$metafeatures)
fit.regression <- function(object,
                           train_performance,
                           train_metafeatures,
                           ...){
  # Training the regressors
  models <- purrr::map(train_performance, function(y, ...){
    ranger::ranger(formula = y ~ .,
                   data = cbind(train_metafeatures, y))
  }, ...)
  res <- list(models = models,
              configurations = names(train_performance),
              recommender_type = 'Regression (RF)')
  class(res) <- class(object)
  return(res)
}



#' Recommend models for new datasets using regression
#'
#' Model recommendation. Issues a single model recommendation per row in `test_metafeatures`.
#'
#' @param object Object obtained from `fit.regression_rf()`.
#' @param test_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#'
#' @return A character string corresponding to the recommended model per row on `test_metafeatures`.
#' @export
#'
#' @examples
#'
#' dat <- get_sample_data()
#' reg <- recommender('regression_rf')
#' reg <- reg %>% fit(dat$train$performance, dat$train$metafeatures, 5, nrounds = 20)
#' reg %>% predict(dat$test$metafeatures)
predict.regression <- function(object, test_metafeatures, score = F){

  predictions <- purrr::map_dfc(object$models, function(x){
    predict(x, test_metafeatures)$predictions
  }) %>% as.matrix
  res <- object$configurations[apply(predictions, 1, which.max)]
  if(score){
    return(dplyr::tibble(score = apply(predictions, 1, max),
                         configuration = res))
  }else{
    class(res) <- c('recommendation', 'character')
    return(res)
  }
}

#' @export
regression_rf_get_residuals <- function(object,
                                        test_metafeatures,
                                        test_performance){
  predictions <- purrr::map_dfc(object$models, function(x){
    predict(x, test_metafeatures)$predictions
  }) %>% as.matrix
  as.matrix(test_performance) - predictions
}


