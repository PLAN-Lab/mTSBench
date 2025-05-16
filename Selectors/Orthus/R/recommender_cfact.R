#' Fit cFact model recommender
#'
#'Model recommender training. This model separates the configurations using clustering and then approximates each clusters' performance matrix using SVD and uses Random Forest to train regression models to get U from metafeatures.
#' @param train_performance A data frame or matrix corresponding one row per dataset and one column per configuration, with performance values on each cell.
#' @param train_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#' @param n_factors Number of factors to use for the SVD decomposition.
#' @param ... Extra parameters to pass to the Random Forest model.
#'
#' @return A list of lists containing three elements each: 1.`models`, one Random Forest per `n_factors`, 2. `dv`, the matrix multiplication of `DV'`,
#' used internally for prediction and 3. `configurations`, the names of the models, used for prediction.
#' @export
#'
#' @examples
#'
#' dat <- get_data(train_test_split = T)
#' cfact <- recommender('cfact')
#' cfact <- cfact %>% fit(dat$train$performance, dat$train$metafeatures, 5)
#' cfact %>% predict(dat$test$metafeatures)
fit.cfact <- function(object,
                      train_performance,
                      train_metafeatures,
                      n_factors = 2,
                      ...){

  algo_proj <- uwot::umap(train_performance %>% t,
                          n_components = 2,
                          metric = 'manhattan',
                          ret_model = F,
                          scale = 'none',
                          n_neighbors = 2  # Dynamically adjust n_neighbors
  )
  algo_clust <- dbscan::dbscan(x = algo_proj, eps = 1)$cluster
  models <- purrr::map(unique(algo_clust), function(clust_num){
  u_reg <- NA
  while(length(u_reg) == 1){
    tryCatch(expr = u_reg <- recommender('u_regression') %>%
                             fit(train_performance %>% dplyr::select(which(algo_clust == clust_num)),
                                 train_metafeatures),
             error = function(e){
               message('Detected error with the internal SVD computations. Trying again.')
               NA
    })
  }
    u_reg
  })
  res <- list(models = models,
              cluster_indices = algo_clust,
              configurations = names(train_performance),
              recommender_type = 'cFact')
  class(res) <- class(object)
  return(res)
}



#' Recommend models for new datasets using cFact
#'
#' Model recommendation. Issues a single model recommendation per row in `test_metafeatures`.
#'
#' @param object Object obtained from `fit.cfact()`.
#' @param test_metafeatures A a data frame containing the metafeatures of each dataset as columns.
#'
#' @return A character string corresponding to the recommended model per row on `test_metafeatures`.
#' @export
#'
#' @examples
#'
#' dat <- get_data(train_test_split = T)
#' cfact <- recommender('cfact')
#' cfact <- cfact %>% fit(dat$train$performance, dat$train$metafeatures, 5)
#' cfact %>% predict(dat$test$metafeatures)
predict.cfact <- function(object, test_metafeatures){
  predictions <- purrr::map(object$models, function(internal_recommender){
    internal_recommender %>% predict(test_metafeatures, score = T)
  })
  configurations <- suppressMessages(purrr::map_dfc(predictions, 'configuration') %>% as.matrix)
  columns <- suppressMessages(apply(purrr::map_dfc(predictions, 'score'), 1, which.max))
  res <- purrr::map_chr(1:nrow(configurations), function(x){
    configurations[x, columns[x]]
  })
  class(res) <- c('recommendation', 'character')
  res
}


