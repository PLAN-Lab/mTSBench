#' @export
get_data <- function(providers = "all",
                     metric = 'affiliation_pr_auc',
                     mfs_metaod = T,
                     mfs_catch22 = T,
                     mfs_scaled = F,
                     train_test_split = F,
                     seed = NULL) {
  #Performance
  perf <- get_precomputed_performance(providers = providers, metric = metric)
  perf <- perf %>% dplyr::arrange(dataset)
  datasets <- perf$dataset
  perf <- perf %>% dplyr::select(-dataset)
  if (!is.null(seed))
    set.seed(seed)
  perf_split <- rsample::initial_split(perf)
  perf_train <- rsample::training(perf_split)
  perf_test <- rsample::testing(perf_split)
  #Metafeatures
  mfs <-
    get_precomputed_metafeatures(
      providers = providers,
      include_metaod = mfs_metaod,
      include_c22 = mfs_catch22,
      scale = mfs_scaled
    )
  mfs$dataset <-
    paste(mfs$provider, mfs$instance, mfs$node, sep = '-')
  mfs <- mfs %>% dplyr::filter(dataset %in% datasets)
  mfs <- mfs[match(datasets, mfs$dataset), ] %>%
    dplyr::select(-provider,-instance,-node,-dataset,-scaled_data)
  mfs_train <- mfs %>%
    dplyr::slice(perf_split$in_id)
  mfs_test <- mfs %>%
    dplyr::slice(-perf_split$in_id)
  #Result
  if (train_test_split) {
    return(list(
      train = list(performance = perf_train,
                   metafeatures = mfs_train),
      test = list(performance = perf_test,
                  metafeatures = mfs_test),
      datasets = datasets,
      configurations = names(perf_train)
    ))
  } else{
    return(
      list(
        performance = perf,
        metafeatures = mfs,
        datasets = datasets,
        configurations = names(perf)
      )
    )
  }
}

#' @export
get_data_leave_one_provider_out <- function(providers = "all",
                                            metric = 'affiliation_pr_auc',
                                            mfs_metaod = T,
                                            mfs_catch22 = T,
                                            mfs_scaled = F,
                                            seed = NULL) {
  perf <-
    get_precomputed_performance(providers, metric = metric) %>% tidyr::nest(-dataset)
  names(perf)[2] <- 'performance'
  mfs <-
    get_precomputed_metafeatures(
      providers = providers,
      include_metaod = mfs_metaod,
      include_c22 = mfs_catch22,
      scale = mfs_scaled
    )
  mfs$dataset <-
    paste(mfs$provider, mfs$instance, mfs$node, sep = '-')
  mfs <-
    mfs %>% dplyr::select(-provider,-instance,-node) %>% tidyr::nest(-dataset)
  names(mfs)[2] <- 'metafeatures'
  joined <- dplyr::left_join(perf, mfs, by = 'dataset')
  joined$provider <-
    purrr::map_chr(strsplit(joined$dataset, split = '-'), 1)
  if (!is.null(seed))
    set.seed(seed)
  list(
    data = rsample::group_vfold_cv(data = joined, group = provider),
    providers = joined$provider
  )
}

#' @export
get_data_stratified_kfolds_cv <- function(providers = "all",
                                          metric = 'affiliation_pr_auc',
                                          mfs_metaod = T,
                                          mfs_catch22 = T,
                                          mfs_scaled = F,
                                          k = 10,
                                          repeats = 3,
                                          seed = NULL) {
  perf <-
    get_precomputed_performance(providers, metric = metric) %>% tidyr::nest(-dataset)
  names(perf)[2] <- 'performance'
  mfs <-
    get_precomputed_metafeatures(
      providers = providers,
      include_metaod = mfs_metaod,
      include_c22 = mfs_catch22,
      scale = mfs_scaled
    )
  mfs$dataset <-
    paste(mfs$provider, mfs$instance, mfs$node, sep = '-')
  mfs <-
    mfs %>% dplyr::select(-provider,-instance,-node) %>% tidyr::nest(-dataset)
  names(mfs)[2] <- 'metafeatures'
  joined <- dplyr::left_join(perf, mfs, by = 'dataset')
  joined$provider <-
    purrr::map_chr(strsplit(joined$dataset, split = '-'), 1)
  if (!is.null(seed))
    set.seed(seed)
  list(
    data = rsample::vfold_cv(
      data = joined,
      strata = provider,
      v = k,
      repeats = repeats
    ),
    providers = joined$provider
  )
}


#' @export
get_data_mc_cv <- function(providers = "all",
                           metric = 'affiliation_pr_auc',
                           mfs_metaod = T,
                           mfs_catch22 = T,
                           mfs_scaled = F,
                           prop = 0.75,
                           times = 25,
                           seed = NULL) {
  perf <-
    get_precomputed_performance(providers, metric = metric) %>% tidyr::nest(-dataset)
  names(perf)[2] <- 'performance'
  mfs <-
    get_precomputed_metafeatures(
      providers = providers,
      include_metaod = mfs_metaod,
      include_c22 = mfs_catch22,
      scale = mfs_scaled
    )
  mfs$dataset <-
    paste(mfs$provider, mfs$instance, mfs$node, sep = '-')
  mfs <-
    mfs %>% dplyr::select(-provider,-instance,-node) %>% tidyr::nest(-dataset)
  names(mfs)[2] <- 'metafeatures'
  joined <- dplyr::left_join(perf, mfs, by = 'dataset')
  joined$provider <-
    purrr::map_chr(strsplit(joined$dataset, split = '-'), 1)
  if (!is.null(seed))
    set.seed(seed)
  list(
    data = rsample::mc_cv(
      data = joined,
      prop = prop,
      times = times
    ),
    providers = joined$provider
  )
}

#' @export
get_train_perf_from_split <- function(dat_split) {
  training <- rsample::training(dat_split)
  training$performance %>% dplyr::bind_rows()
}

#' @export
get_train_mfs_from_split <- function(dat_split) {
  training <- rsample::training(dat_split)
  train_metafeatures <-
    training$metafeatures %>% dplyr::bind_rows() %>% dplyr::select(-scaled_data)
}

#' @export
get_test_perf_from_split <- function(dat_split) {
  testing <- rsample::testing(dat_split)
  testing$performance %>% dplyr::bind_rows()
}

#' @export
get_test_mfs_from_split <- function(dat_split) {
  testing <- rsample::testing(dat_split)
  testing$metafeatures %>% dplyr::bind_rows() %>% dplyr::select(-scaled_data)
}
