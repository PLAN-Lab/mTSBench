#' Evaluate Recommender Output
#'
#' This function takes the output of any recommender and evaluates the percentile of performance in which the output recommendations fall.
#'
#' @param recs A character vector. Output of any of the `predict()` functions output by any `recommender` object.
#' @param perfs A data frame corresponding to one row per dataset and one column per configuration, with performance values on each cell.
#'
#' @return A data frame with four columns: `recommendation` (the recommended configuration), `performance` (the performance value of the selected option), `performance_max` (the maximum possible performance for that dataset), and `percentile` (the percentile of performance in which the recommendation falls in).
#' @export
#'
#' @examples
#'
#' dat <- get_data()
#' gb <- recommender('global_best')
#' gb <- gb %>% fit(dat$train$performance, dat$train$metafeatures, median = T)
#' gb %>% predict(dat$test$metafeatures)
#' recs <- gb %>% predict(dat$test$metafeatures)
#' perfs <- dat$test$performance
#' evaluate_recommendations(recs, perfs)
evaluate_recommendations <- function(recs,
                                     perfs){
# Performance of recommendation, max performance, position
  purrr::map2_dfr(recs, 1:length(recs), function(x, dataset){
    p <- perfs[dataset,]
    index <- which(names(p) == x)
    dplyr::tibble(recommendation = x,
                  performance = p[1,index] %>% as.numeric,
                  performance_max =  max(unlist(p)),
                  percentile = round(x = length(which(as.numeric(p[1,index]) >= unlist(p)))/ncol(p), digits = 2)*100)
  })
}
