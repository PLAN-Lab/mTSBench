get_percentile_matrix <- function(res){
  res %>%
    bind_rows_manual() %>%
    group_by(recommender, dataset) %>%
    group_map(., function(x,...){
      x$index <- 1:nrow(x)
      x
    },.keep = T) %>%
    bind_rows %>%
    select(recommender, percentile, dataset, index) %>%
    pivot_wider(data = ., names_from = 'recommender', values_from = 'percentile')
}
