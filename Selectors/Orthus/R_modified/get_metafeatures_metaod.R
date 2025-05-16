#' @export
get_metafeatures_metaod <- function(x,
                                    scale_data = F,
                                    show_message = T){
  if(show_message){
    message('Calculating the MetaOD metafeatures needs a valid python installation and several dependencies. Please use the install_metaod_dependencies() function to manage them. Call this function with show_messages = FALSE to hide this message.')
  }
  reticulate::source_python(system.file("python", "metaod_metafeatures.py", package = "autoad"))
  if(scale_data) x <- purrr::map_dfc(x, normalize)
  raw_output <- generate_meta_features(x %>% as.matrix)
  df_names <- unlist(raw_output[[2]])
  df <- data.frame(t(data.frame(unlist(raw_output[[1]]))))
  names(df) <- df_names
  rownames(df) <- NULL
  df
}
