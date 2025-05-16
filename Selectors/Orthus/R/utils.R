rescale <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

normalize <- function(x, na.rm = T){
  return((x- mean(x)) /(sd(x)))
}

robust_normalize <- function(x, na.rm = T){
  return((x- median(x)) /(mad(x)))
}

sample_and_hold_vector <- function(x){
  num_values <- which(!is.na(x))
  # Fill first values
  if(num_values[1] > 1){
    x[1:num_values[1] -1] <- x[num_values[1]]
  }
  missing_values <- which(is.na(x))
  if(length(missing_values) > 0){
    for(i in 1:length(missing_values)){
      missing_position <- missing_values[i]
      value_to_use <- x[num_values[max(which(num_values < missing_position))]]
      x[missing_values[i]] <- value_to_use
    }
  }
  x
}

sample_and_hold_controller <- function(x){
  if('numeric' %in% class(x)){
    return(sample_and_hold_vector(x))
  }else{
    return(x)
  }
}

sample_and_hold_df <- function(df){
  purrr::map_dfc(df, sample_and_hold_controller)
}

get_provider_names <- function(){
  c('Benchmark', 'BGP', 'SMD', 'Water')
}

#' @export
fit <- function(object, ...){
  generics::fit(object, ...)
}

#' @export
predict <- function(object, ...){
  stats::predict(object, ...)
}

#' @export
prepare_parallelization <- function(){
  isRStudio <- Sys.getenv("RSTUDIO") == "1"
  if(isRStudio){
    future::plan(future::multisession)
  }else{
    future::plan(future::multicore)
  }
  NULL
}

bind_rows_manual <- function(x){
  res <- x[[1]]
  for(i in 2:length(x)){
    res <- rbind(res, x[[i]])
  }
  res
}
