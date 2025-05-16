#' Fit Global Best model recommender
#'
#' Baseline recommender: selects the configuration with best mean / median out of all and always recommends it.
#'
#' @param train_performance A data frame or matrix corresponding one row per dataset and one column per configuration, with performance values on each cell.
#' @param train_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#' @param median Logical. Whether to use median (`TRUE`) or mean (`FALSE`) to select a model. Defaults to `TRUE`.
#'
#' @return A list containing one element: `recommendation`, the selected configuration.
#' @export
#'
#' @examples
#'
#' dat <- get_sample_data(train_test_split = T)
#' gb <- recommender('global_best')
#' gb <- gb %>% fit(dat$train$performance, dat$train$metafeatures, median = T)
#' gb %>% predict(dat$test$metafeatures)
fit.global_best <- function(object,
                            train_performance,
                           train_metafeatures = NULL,
                           median = T,...){
  if(median){
    values <- apply(train_performance, 2, median)
  }else{
    values <- apply(train_performance, 2, mean)
  }
  res <- list(recommendation = names(train_performance)[which.max(values)],
              recommender_type = 'Global Best')
  class(res) <- class(object)
 return(res)
}


#' Recommend models for new datasets using globalbest
#'
#' Model recommendation. Issues a single model recommendation per row in `test_metafeatures`.
#'
#' @param object Object obtained from `globalbest_fit()`.
#' @param test_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#'
#' @return A character string corresponding to the recommended model per row on `test_metafeatures`.
#' @export
#'
#' @examples
#'
#' dat <- get_sample_data(train_test_split = T)
#' gb <- recommender('global_best')
#' gb <- gb %>% fit(dat$train$performance, dat$train$metafeatures, median = T)
#' gb %>% predict(dat$test$metafeatures)
predict.global_best <- function(object,
                               test_metafeatures){
  res <- rep(object$recommendation,
             times = nrow(test_metafeatures))
  class(res) <- c('recommendation', 'character')
  res
}

