#' @export
get_metafeatures <- function(x,
                             include_metaod = TRUE,
                             include_catch22 = TRUE,
                             scale_data = FALSE,
                             process_metaod_vars = TRUE,
                             time_var = NULL,
                             show_message = TRUE){
  # Checks
  if(!include_metaod & !include_catch22){
    stop('You have to select at least one metafeature set (metaOD or catch22).')
  }
  # Calculations
  if(include_metaod){
    if(!is.null(time_var)){
      x_metaod <- x %>%
        dplyr::select(-dplyr::any_of(time_var))
    }else{
      x_metaod <- x
    }
    mfs_metaod <- get_metafeatures_metaod(x_metaod, scale_data, show_message = show_message)
    if(process_metaod_vars){
      mfs_metaod <- sanitize_metafeatures_metaod(mfs_metaod)
    }
  }
  if(include_catch22){
    mfs_c22 <- get_metafeatures_catch22(x, time_var, scale_data)
  }
  # Return
  if(include_metaod & !include_catch22) return(mfs_metaod)
  if(!include_metaod & include_catch22) return(mfs_c22)
  if(include_metaod & include_catch22) return(cbind(mfs_metaod, mfs_c22))
}
