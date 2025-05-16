#' @export
fit.orthus_tunable <- function(object,
                              train_performance,
                              train_metafeatures,
                              eps,
                              minpts){
  local_head <- recommender('cfact') %>%
    fit(train_performance, train_metafeatures)
  global_head <- recommender('u_regression') %>%
    fit(train_performance, train_metafeatures, n_factors = 2)
  res <- list(local_head = local_head,
              global_head = global_head,
              train_metafeatures = train_metafeatures,
              recommender_type = 'Orthus',
              epsilon = eps,
              min_pts = minpts)
  class(res) <- class(object)
  res
}

#' @export
predict.orthus_tunable <- function(object, test_metafeatures){
  train_metafeatures <- object$train_metafeatures
  train_performance <- object$train_performance
  type_reference <- tidyr::tibble(index = 1:(nrow(train_metafeatures) + nrow(test_metafeatures)),
                                  type = c(rep('train', times = nrow(train_metafeatures)),
                                           rep('test', times = nrow(test_metafeatures))))
  metafeatures <- rbind(train_metafeatures, test_metafeatures)
  # Project train + test data
  scaler <- caret::preProcess(x = metafeatures, method = c('zv', 'center', 'scale'))
  scaled_mfs <- predict(scaler, newdata = metafeatures)
  set.seed(7777)
  complete_data_proj <- uwot::umap(scaled_mfs,
                                   n_components = 10,
                                   metric = 'manhattan',
                                   ret_model = F,
                                   scale = 'none')
  # Cluster projected data
  data_clustering <- dbscan::dbscan(x = complete_data_proj, eps = object$epsilon, minPts = object$min_pts)
  # Create regression models for clusters in which there are points from the test data
  test_clusters <- data_clustering$cluster[type_reference %>% filter(type == 'test') %>% pull(index)]
  recommenders <- map(unique(test_clusters), function(clust_num){
    train_metafeatures_rows <- which(data_clustering$cluster[type_reference %>% filter(type == 'train') %>% pull(index)] == clust_num)
    if(length(train_metafeatures_rows) > 10){
      # With enough samples, we train a regression recommender
      return(object$local_head)
    }else{
      # We don't have enough samples to train a recommender, just return the global recommender
      return(object$global_head)
    }
  })
  names(recommenders) <- paste0('C', unique(test_clusters))
  # Predict, per test point, with corresponding recommender
  map_dfr(unique(test_clusters), function(clust_num){
    samples_index <- which(test_clusters == clust_num)
    dplyr::tibble(index = samples_index,
                  recommendation = recommenders[[paste0('C', clust_num)]] %>% predict(test_metafeatures %>% slice(samples_index)))
  }) %>%
    arrange(index) %>%
    pull(recommendation) -> res
  class(res) <- c('recommendation', 'character')
  res
}
