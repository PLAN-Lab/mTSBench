#' @export
fit.metaod <- function(object,
                       train_performance,
                       train_metafeatures,
                       n_factors = 5L,
                       show_message = TRUE,
                       ...){
  if(show_message){
    message('Using the MetaOD recommender needs a valid python installation and several dependencies. Please use the install_metaod_dependencies() function to manage them. Call this function with show_messages = FALSE to hide this message.')
  }
  reticulate::source_python(system.file("python", "metaod_recommender.py", package = "ADRecommender"))
  recommender <- MetaODClass(train_performance = train_performance %>% as.matrix,
                             verbose = T,
                             n_factors = n_factors)
  mfs_scaler <- caret::preProcess(train_metafeatures,
                                  method = 'range')
  train_metafeatures <- predict(mfs_scaler, train_metafeatures)
  recommender <- recommender$train(meta_features = train_metafeatures %>% as.matrix)
  res <- list(recommender = recommender,
              metafeatures_scaler = mfs_scaler,
              configurations = names(train_performance),
              recommender_type = 'MetaOD')
  class(res) <- class(object)
  res
}


#' @export
predict.metaod <- function(object,
                           test_metafeatures){
  test_metafeatures <- predict(object$metafeatures_scaler, test_metafeatures)
  predictions <- object$recommender$predict(test_metafeatures)
  res <- object$configurations[apply(predictions, 1, which.max)]
  class(res) <- c('recommendation', 'character')
  res
}
