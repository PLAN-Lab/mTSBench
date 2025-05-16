get_data_from_split <- function(dat_split){
  training <- rsample::training(dat_split)
  train_performance <- training$performance %>% bind_rows_manual()
  train_metafeatures <- training$metafeatures %>% bind_rows_manual() %>% select(-scaled_data)
  testing <- rsample::testing(dat_split)
  test_performance <- testing$performance %>% bind_rows_manual()
  test_metafeatures <- testing$metafeatures %>% bind_rows_manual() %>% select(-scaled_data)
  list(train = list(performance = train_performance,
                    metafeatures= train_metafeatures),
       test = list(performance = test_performance,
                   metafeatures = test_metafeatures))
}
