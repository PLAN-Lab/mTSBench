#' Fit Nearest Neighbor model recommender
#'
#' Baseline recommender: selects the best configuration of the closest dataset in the metafeature space.
#'
#' @param train_performance A data frame or matrix corresponding one row per dataset and one column per configuration, with performance values on each cell.
#' @param train_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#' @param scale_data Logical. Whether to calculate the zscore, center and scale the meta-feature space. Defaults to `TRUE`.
#'
#' @return A list containing one element: `recommendation`, the selected configuration.
#' @export
#'
#' @examples
#'
#' dat <- get_data(train_test_split = TRUE)
#' nn <- recommender('nearest_neighbor')
#' nn <- nn %>% fit(dat$train$performance, dat$train$metafeatures)
#' nn %>% predict(dat$test$metafeatures)
fit.nearest_neighbor <- function(object,
                            train_performance,
                            train_metafeatures,
                            scale_data = TRUE,
                            ...){
  # The projection and metafeatures are stored to be used in prediction,
  # alongside the best model per dataset
  best_models <- names(train_performance)[apply(train_performance, 1, which.max)]
  res <- list(reference_space = train_metafeatures,
              best_configuration = best_models,
              scale_data = scale_data,
              recommender_type = 'Nearest Neighbor')
  class(res) <- class(object)
  return(res)
}


#' Recommend models for new datasets using Nearest Neighbor
#'
#' Model recommendation. Issues a single model recommendation per row in `test_metafeatures`. Each dataset is compared against the training
#' metafeatures and their nearest neighbor is obtained. The configuration with maximum performance in the nearest neighbor is recommended.
#'
#' @param object Object obtained from `fit.nearest_neighbor()`.
#' @param test_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#'
#' @return A character string corresponding to the recommended model per row on `test_metafeatures`.
#' @export
#'
#' @examples
#'
#' dat <- get_data(train_test_split = TRUE)
#' nn <- recommender('nearest_neighbor')
#' nn <- nn %>% fit(dat$train$performance, dat$train$metafeatures)
#' nn %>% predict(dat$test$metafeatures)
predict.nearest_neighbor <- function(object,
                                test_metafeatures){
  if(object$scale_data){
      constant_features <- names(object$reference_space)[which(purrr::map_lgl(object$reference_space, function(x) length(unique(x)) == 1))]
      if(length(constant_features) > 0){
        test_metafeatures <- test_metafeatures %>%
          dplyr::select(-dplyr::all_of(constant_features))
      }
      scaler <- caret::preProcess(object$reference_space, method = c('zv', 'center', 'scale'))
      object$reference_space <- predict(scaler, object$reference_space)
      test_metafeatures <- predict(scaler,
                                   newdata = test_metafeatures)
  }
  joined_data <- rbind(object$reference_space,
                       test_metafeatures)
  nns <- FNN::get.knnx(data = joined_data %>% dplyr::slice(1:nrow(object$reference_space)) %>% as.matrix(),
                       query = joined_data %>% dplyr::slice((nrow(object$reference_space)+1):nrow(joined_data)) %>% as.matrix(),
                       k = 1)$nn.index %>% as.numeric
  res <- object$best_configuration[nns]
  class(res) <- c('recommendation', 'character')
  res
}

