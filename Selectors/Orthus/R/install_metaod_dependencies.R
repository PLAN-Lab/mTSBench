#' @export
install_metaod_dependencies <- function(method = "auto", conda = "auto", ...) {
  reticulate::py_install(packages = c("pandas",
                                      "numpy",
                                      "scikit-learn",
                                      "scipy",
                                      "pyod"),
                         method = method,
                         conda = conda, ...)
}
