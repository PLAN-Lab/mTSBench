# 1) First, detach it if it’s loaded in this session:
if ("package:ADRecommender" %in% search()) {
  detach("package:ADRecommender", unload = TRUE)
}

# 2) Then remove it from your library:
remove.packages("ADRecommender")

# 1) Point to your package root
setwd("/Volumes/LaCie/mTSB/Orthus")  

# 2) Re-install
# (this will re-build the NAMESPACE from your roxygen comments, compile, and install)
devtools::install()

# 3) Load it
library(ADRecommender)

