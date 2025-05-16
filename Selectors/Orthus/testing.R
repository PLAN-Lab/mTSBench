library(ADRecommender)
library(readr)

dat <- get_data(mfs_metaod = FALSE, mfs_catch22 = TRUE)
performance <- dat$performance
performance <- dat$performance[2:94,101:109]
metafeatures <- dat$metafeatures[2:94,]
rec <- recommender('orthus') %>% fit(auc_matrix_numeric, metafeatures)
rec %>% predict(new_metafeatures) #new_metafeatures)
#new_data <- readr::read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
#new_metafeatures <- get_metafeatures(new_data, include_metaod = FALSE, include_catch22 = TRUE, time_var = 'date')



summary(performance[2:94, 100:109])  # Failing
summary(performance[2:94, 499:507])  # Working

result <- get_precomputed_performance()
print(result)
library(readr)
auc_matrix <- read_csv("/Users/xiazhou/Desktop/ADRecommender/auc_matrix.csv")  #("auc_matrix.csv") #
meta_matrix <- read_csv("/Users/xiazhou/Desktop/ADRecommender/train_meta_features.csv") #("train_meta_features.csv") # 
meta_matrix_numeric <- meta_matrix[, -ncol(meta_matrix)] # exclude the dataset column



# Exclude the first column
auc_matrix_numeric <- auc_matrix[, -1]

summary(auc_matrix_numeric)
# Check if any rows in auc_matrix_numeric do not contain any 0s
no_zeros_rows <- apply(auc_matrix_numeric, 1, function(row) all(row != 0))

# View the result
no_zeros_indices <- which(no_zeros_rows)  # Indices of rows without any 0s
no_zeros_indices

# Check the count of such rows
cat("Number of rows without any 0s:", length(no_zeros_indices), "\n")

# Optionally, view these rows
auc_matrix_numeric[no_zeros_indices, ]


# Subset the first 5 rows
#auc_matrix_subset <- auc_matrix[1:20, ]
#meta_matrix_subset <- meta_matrix[1:20, ]
#auc_matrix_numeric <- auc_matrix_numeric[,1:4]
rec_our <- recommender('orthus') %>% fit(auc_matrix_numeric,meta_matrix_numeric, eps= 1, minPts=2)
rec_our %>% predict(new_metafeatures) 
test_data <- read_csv("water.test.csv")  #Downloads/ADRecommender/
test_metafeatures <- get_metafeatures(test_data, include_metaod = FALSE, include_catch22 = TRUE, time_var = 'timestamp')
rec_our %>% predict(new_metafeatures) 
library(uwot)

# Reduce dimensions for auc_matrix and meta_matrix
reduced_auc <- umap(auc_matrix, n_neighbors = 2)
reduced_meta <- umap(meta_matrix, n_neighbors = 2)

# Fit the recommender using reduced data
rec <- recommender('orthus') %>% fit(reduced_auc, reduced_meta, n_factors=5)
rec <- recommender('orthus') %>% fit(reduced_auc, reduced_meta)


# Check for zero-variance columns
apply(performance, 2, var)

# Remove zero-variance or near-zero-variance columns
performance <- performance[, apply(performance, 2, var) > 0.01]



test_function <- algorithm_parser[[algorithm]]
print(test_function)
test_function(rec[i])  # Manually run it


