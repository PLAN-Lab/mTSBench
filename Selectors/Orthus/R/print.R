print.recommendation <- function(rec) {
  for (i in 1:length(rec)) {
    algorithm <- strsplit(rec[i], '_', fixed = T)[[1]][1]
    if (!is.null(algorithm_parser[[algorithm]])) {
      print(paste0('Dataset ', i, ': ', algorithm_parser[[algorithm]](rec[i])))
    } else {
      print(paste0('Dataset ', i, ': Unknown algorithm (', algorithm, ')'))
    }
  }
}

algorithm_parser <- list()

# Parsers for the old algorithms
algorithm_parser[['rhf']] <- function(x) {
  splitted <- strsplit(x, '-', fixed = T)[[1]]
  paste0('RHF (',
         splitted[2], ' trees, ',
         splitted[3], ' max height, ',
         ifelse(as.logical(splitted[4]),
                yes = 'check duplicates)',
                no = 'do not check duplicates)'))
}

algorithm_parser[['hst']] <- function(x) {
  splitted <- strsplit(x, '-', fixed = T)[[1]]
  paste0('HST (',
         splitted[2], ' trees, ',
         splitted[3], ' max tree depth, ',
         as.numeric(splitted[4]) * 100, '% of data as initial sample)')
}

algorithm_parser[['isolation_forest']] <- function(x) {
  splitted <- strsplit(x, '-', fixed = T)[[1]]
  paste0('Isolation Forest (',
         as.numeric(splitted[3]), '% of data per tree, ',
         splitted[4], ' max tree depth, ',
         splitted[5], ' trees, ',
         splitted[6], ' columns per split)')
}

algorithm_parser[['loda_batch']] <- function(x) {
  splitted <- strsplit(x, '-', fixed = T)[[1]]
  paste0('LODA (batch, ',
         splitted[2], ' bins per histogram, ',
         splitted[3], ' random cuts)')
}

algorithm_parser[['lof']] <- function(x) {
  splitted <- strsplit(x, '-', fixed = T)[[1]]
  paste0('LOF (',
         'Minkowski exponent = ', splitted[2], ', ',
         'leaf size = ', splitted[3],
         ', number of neighbors = ', splitted[4], ')')
}

algorithm_parser[['xstream']] <- function(x) {
  splitted <- strsplit(x, '-', fixed = T)[[1]]
  paste0('xStream (',
         'Random projections of size ', splitted[2], ', ',
         splitted[3], ' chains of depth ', splitted[4], ', ',
         as.numeric(splitted[6]) * 100, '% of data as initial sample)')
}

# Parsers for the new algorithms (no input parsing needed)
algorithm_parser[['copod']] <- function(x) {
  "COPOD (Copula-Based Outlier Detection)"
}
algorithm_parser[['eif']] <- function(x) {
  "EIF (Extended Isolation Forest)"
}
algorithm_parser[['hif']] <- function(x) {
  "HIF (Histogram Isolation Forest)"
}
algorithm_parser[['kmeans']] <- function(x) {
  "K-Means Clustering"
}
algorithm_parser[['knn']] <- function(x) {
  "K-Nearest Neighbors (KNN)"
}
algorithm_parser[['lstm_ad']] <- function(x) {
  "LSTM-AD (Long Short-Term Memory for Anomaly Detection)"
}
algorithm_parser[['robust_pca']] <- function(x) {
  "Robust PCA (Principal Component Analysis)"
}
algorithm_parser[['telemanom']] <- function(x) {
  "Telemanom (Telemetry Anomaly Detection)"
}
algorithm_parser[['torsk']] <- function(x) {
  "TORSK (Time-Order Recurrent State-Space Kernel)"
}

# Export function for recommender printing
#' @export
print.recommender <- function(object) {
  print(paste0('Fit recommender of type ', object$recommender_type))
}
