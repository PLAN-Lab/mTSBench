get_performance_matrix <- function(res){
  res %>%
    bind_rows_manual() %>%
    group_by(recommender, dataset) %>%
    group_map(., function(x,...){
      x$index <- 1:nrow(x)
      x
    },.keep = T) %>%
    bind_rows %>%
    select(recommender, performance, performance_max, dataset, index) %>%
    pivot_wider(data = ., names_from = 'recommender', values_from = 'performance')
}
