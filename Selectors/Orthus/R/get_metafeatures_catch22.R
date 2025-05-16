#' @export
get_metafeatures_catch22 <- function(x,
                                     time_var_name = 'timestamp',
                                     scale_data = F){
  if(is.null(time_var_name)){
    x$timestamp <- 1:nrow(x)
    time_var_name <- 'timestamp'
  }
  # Scale features to unit interval
  if(scale_data){
    timestamp <- x %>% dplyr::select(all_of(time_var_name))
    x <- purrr::map_dfc(x %>% dplyr::select(-time_var_name), normalize) %>% dplyr::mutate(timestamp = timestamp)
  }
  # Calculate base features
  x <- x %>% tidyr::pivot_longer(cols = -dplyr::any_of(time_var_name))
  features <- theft::calculate_features(x,
                                        id_var = 'name',
                                        time_var = time_var_name,
                                        values_var = 'value',
                                        feature_set = 'catch22')
  # Calculate feature summaries
  features %>%
    dplyr::group_by(names) %>%
    dplyr::group_map(., function(y, ...){
      res <- c(min(y$values),
               quantile(y$values, prob = 0.25),
               mean(y$values),
               quantile(y$values, prob = 0.75),
               max(y$values))
      names(res) <- paste(y$names[1], c('min', 'q25', 'mean', 'q75', 'max'), sep ='_')
      res
    }, .keep = T) %>%
    unlist() %>%
    t() %>%
    as.data.frame()
}
