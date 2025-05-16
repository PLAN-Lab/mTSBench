fit_and_evaluate_recommender <- function(rec_type, train_performance, train_metafeatures, test_performance, test_metafeatures){
  recommender(rec_type) %>%
    fit(train_performance = train_performance,
        train_metafeatures = train_metafeatures) %>%
    predict(test_metafeatures) %>%
    evaluate_recommendations(., test_performance)
}

