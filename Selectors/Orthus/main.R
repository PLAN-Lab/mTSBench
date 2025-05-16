# Package management
install_required_packages <- function(){
  print('Installing needed packages')
  packs <- c('here', 'devtools',
             'reticulate', 'theft', 'magrittr',
             'tidyverse', 'furrr', 'logger', 'generics',
             'FNN', 'uwot', 'rsample', 'dbscan', 'caret', 'ranger',
             'httpuv', 'xtable', 'logger', 'cowplot', 'meta')
  installed_packages <- as.character(installed.packages()[,1])
  packages_not_installed <- packs[which(!(packs %in% installed_packages))]
  while(length(packages_not_installed) > 0){
    install.packages(packages_not_installed)
    packages_not_installed <- packs[which(!(packs %in% installed_packages))]
  }
  print('Packages installed')
}
install_required_packages()
library(here)
setwd(here())
setwd("/Volumes/LaCie/mTSB/Orthus/")  # Set the correct package directory
getwd()  # Verify the current working directory

devtools::install()
if(!('scmamp' %in% as.character(installed.packages()[,1]))){
  devtools::install_github('b0rxa/scmamp')
}

# Sourcing the experiment files
logger::log_info('Generating figure 3 in paper_scripts/figures/datasets.pdf')
source(here('paper_scripts', 'figures', 'datasets.R'))
logger::log_info('Generating table 4 of metafeatures comparison:')
source(here('paper_scripts', 'figures', 'metafeatures.R'))
logger::log_info('Generating figure 4 in paper_scripts/figures/recommender_comparison.pdf')
source(here('paper_scripts', 'figures', 'recommender_comparison.R'))
logger::log_info('Generating figure 5 in paper_scripts/figures/uregression_comparison.pdf')
source(here('paper_scripts', 'figures', 'uregression_vectors.R'))
logger::log_info('Generating figure 6 in paper_scripts/figures/ablation_scenario1.pdf and paper_scripts/figures/ablation_scenario2.pdf')
source(here('paper_scripts', 'figures', 'ablation.R'))
