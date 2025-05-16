#' @export
recommender <- function(type){
  return(structure(list(), class = c(type, 'recommender', 'list')))
}

#' @export
print.recommender <- function(x){
  if(length(x) == 0){
    print(paste('Empty recommender of type', class(x)[1]))
  }else{
    print(paste('Fit recommender of type', class(x)[1]))
  }
}

fit.default <- function(object, ...){
  stop('Wrong recommender selected. Available recommenders: u_regression, global_best, direct_regression')
}
