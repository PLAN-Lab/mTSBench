process_metafeatures <- function(train_metafeatures, test_metafeatures, rec_type){
  if(rec_type == c('nearest_neighbor')){
    scaler <- caret::preProcess(train_metafeatures, method = c('zv', 'center', 'scale'))
    train_metafeatures <- predict(scaler, train_metafeatures)
    test_metafeatures <- predict(scaler, test_metafeatures)
  }
  if(rec_type %in% c('metaod')){
    scaler <- caret::preProcess(train_metafeatures, method = c('zv', 'range'))
    train_metafeatures <- predict(scaler, train_metafeatures)
    test_metafeatures <- predict(scaler, test_metafeatures)
    # Sanity checks
    if(any(is.na(unlist(train_metafeatures))) | any(is.infinite(unlist(train_metafeatures)))){
      logger::log_error('Bad values on train metafeatures during ', rec_type, ' on split ', split_num)
      browser()
    }
    if(any(is.na(unlist(test_metafeatures))) | any(is.infinite(unlist(test_metafeatures)))){
      logger::log_error('Bad values on test metafeatures during ', rec_type, ' on split ', split_num)
      browser()
    }
    min_value <- -3.4028235e+38
    max_value <- 3.4028235e+38
    if(any(unlist(train_metafeatures) < min_value) | any(unlist(train_metafeatures) > max_value)){
      logger::log_error('Overflow on train metafeatures during ', rec_type, ' on split ', split_num)
      train_metafeatures[train_metafeatures < min_value] <- min_value/10
      train_metafeatures[train_metafeatures < max_value] <- max_value/10
    }
    if(any(unlist(test_metafeatures) < min_value) | any(unlist(test_metafeatures) > max_value)){
      logger::log_warn('Overflow on test metafeatures during ', rec_type, ' on split ', split_num)
      test_metafeatures[test_metafeatures < min_value] <- min_value/10
      test_metafeatures[test_metafeatures < max_value] <- max_value/10
    }
  }
  list(train = train_metafeatures,
       test = test_metafeatures)
}
