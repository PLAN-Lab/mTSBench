#' Get Precomputed Metafeatures
#'
#' Get a data frame of precomputed metafeatures for almost any of the data providers we use. The code to generate them is in
#' `data-raw/generate_metafeatures_cache.R`.
#'
#' @param providers `all` or any combination of `BGP`, `Benchmark`, `SMD` and `Water`. Defaults to `all`.
#' @param include_metaod Logical. Add metaod metafeatures. Defaults to `TRUE`.
#' @param include_c22 Logical. Add catch22 metafeatures. Defaults to `TRUE`.
#' @param sanitize Logical. Sanitize the metafeatures, discarding high moments and correcting missing / infinite values to appropriate ones. Defaults to `TRUE`.
#' @param scale Logical. Transform the metafeatures to same units, rescaling them to 0-1. Defaults to `TRUE`.
#'
#' @return Data frame with columns `provider`, `instance`, `node` and the selected metafeatures associated with that row's dataset.
#' @export
get_precomputed_metafeatures <-
  function(providers = 'all',
           include_metaod = T,
           include_c22 = T,
           sanitize = T,
           scale = T) {
    # Input validation ----
    if(!(providers %in% c('all', get_provider_names()))){
      stop('providers must be one of "all", "Benchmark", "BGP", "SMD" or "Water"')
    }
    # Code ----
    if(providers == 'all') providers <- get_provider_names()
    res <- ADRecommender:::metafeatures[providers] %>% dplyr::bind_rows()
    res$metafeatures <- vector(mode = "list", length = nrow(res))
    if (include_metaod & !include_c22) {
      res$metafeatures <- res$metafeatures_metaod
    }
    if (!include_metaod & include_c22) {
      res$metafeatures <- res$metafeatures_catch22
    }
    if (include_metaod & include_c22) {
      res$metafeatures <-
        purrr::map2(res$metafeatures_metaod,
                    res$metafeatures_catch22,
                    function(x, y) {
          cbind(x, y)
        })
    }
    res <- res %>%
      dplyr::select(-metafeatures_metaod,
             -metafeatures_catch22) %>%
      tidyr::unnest(metafeatures)
    if(any(c(sanitize, scale))){
      left_side <- res %>% dplyr::select(provider, instance, node)
      right_side <- res %>% dplyr::select(-provider, -instance, -node)
      if(sanitize){
       right_side <- sanitize_metafeatures_metaod(right_side)
      }
      if(scale){
        right_side <- purrr::map_dfc(right_side, function(x){
          if(length(unique(x)) == 1){
            return(rep(0, times = length(x)))
          }else{
            return((x - min(x))/(max(x) - min(x)))
          }
        })
      }
      res <- cbind(left_side, right_side)
    }
   res
  }
