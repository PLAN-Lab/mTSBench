sanitize_metafeatures_metaod <- function(x){
  x <- x %>%
    dplyr::select(-dplyr::any_of(dplyr::starts_with('moment_5')),
                  -dplyr::any_of(dplyr::starts_with('moment_6')),
                  -dplyr::any_of(dplyr::starts_with('moment_7')),
                  -dplyr::any_of(dplyr::starts_with('moment_8')),
                  -dplyr::any_of(dplyr::starts_with('moment_9')),
                  -dplyr::any_of(dplyr::starts_with('moment_10')))
  if('quant_coeff_disp' %in% names(x)){
    x$quant_coeff_disp[is.na(x$quant_coeff_disp)|is.infinite(x$quant_coeff_disp)] <- 0
  }
  if('normality_p_skewness' %in% names(x)){
    x$normality_p_skewness[is.na(x$normality_p_skewness)|is.infinite(x$normality_p_skewness)] <- 0
  }
  if('normality_p_kurtosis' %in% names(x)){
    x$normality_p_kurtosis[is.na(x$normality_p_kurtosis)|is.infinite(x$normality_p_kurtosis)] <- -3
  }
  x
}
