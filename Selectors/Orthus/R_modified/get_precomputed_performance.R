#' Get Precomputed Performances
#'
#' This function returns precomputed Precision Recall AUC for a large combination of datasets and configurations. It can be
#' used to quickly evaluate recommender functions.
#'
#' @param providers `all` or any combination of `BGP`, `Benchmark`, `SMD` and `Water`. Defaults to `all`.
#' @param wide Logical. Whether to return the performance data frame in wide or long format. Defaults to `TRUE`.
#' @param metric One of `all`, `affiliation_pr_auc`, `pr_auc` or `roc`. Which metric to return. Defaults to `affiliation_prauc`. If `all` is selected, `wide` cannot be `TRUE`.
#'
#' @return A data frame of wide format with the selected metric.
#' @export
get_precomputed_performance <- function(providers = 'all',
                                        wide = T,
                                        metric = 'affiliation_pr_auc'){
  # Input validation ----
  if(!(providers %in% c('all', get_provider_names()))){
    stop('providers must be one of "all", "Benchmark", "BGP", "SMD" or "Water"')
  }
  if(!(metric %in% c('all', 'affiliation_pr_auc', 'pr_auc', 'roc'))){
    stop('metric must be one of "all", "affiliation_pr_auc", "pr_auc" or "roc"')
  }
  if(metric == 'all' & wide){
    stop('If wide = TRUE, metric cannot be "all"')
  }
  # Code ----
  if(providers == 'all') providers <- get_provider_names()
  cols_to_keep <- c('dataset', 'algorithm', 'configuration', metric)
  res <- ADRecommender:::performance %>%
    dplyr::filter(prov %in% providers) %>%
    dplyr::select(dplyr::all_of(cols_to_keep))
  if(wide){
    res <- res %>%
      tidyr::pivot_wider(id_cols = 'dataset',
                         names_from = c('algorithm', 'configuration'),
                         values_from = dplyr::all_of(metric))
  }
  res
}
