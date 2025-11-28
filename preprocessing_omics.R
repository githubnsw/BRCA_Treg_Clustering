# ============================================================
# Data Preprocessing Pipeline for miRNA Expression Data
# Author: [Your Name]
# Description: Filtering, imputation, and normalization of miRNA data
# ============================================================

library(data.table)
library(dplyr)
library(impute)

# -------------------- Functions --------------------

#' Min-Max Normalization
#' @param data A data frame with numeric values
#' @return Normalized data frame with values scaled to [0, 1]
minmax_normalize <- function(data) {
  normalized_data <- as.data.frame(lapply(data, function(x) {
    if (min(x, na.rm = TRUE) == max(x, na.rm = TRUE)) {
      return(rep(0, length(x)))
    }
    (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  }))
  rownames(normalized_data) <- rownames(data)
  return(normalized_data)
}

# -------------------- User Parameters --------------------
input_file    <- "data/mirna_validation_only.csv"
output_file   <- "results/filtered_mirna_validation_only.csv"

missing_cut   <- 0.2    # Missing value threshold (remove rows with >20% missing)
neighbors     <- 5      # Number of neighbors for KNN imputation
apply_minmax  <- TRUE   # Apply Min-Max normalization
apply_impute  <- FALSE  # Apply KNN imputation (set FALSE for RNA data)
top_features  <- 2000   # Number of top variable features to select
# ---------------------------------------------------------

# -------------------- Main Pipeline --------------------

# Load data
x <- data.frame(fread(input_file), row.names = 1)
cat("Initial Data Dimensions:", dim(x), "\n")

# 1. Filter rows with excessive missing values or zeros
cat("Filtering rows with missing values or excessive zeros...\n")
filtered_data <- x[rowSums(is.na(x)) < ncol(x) * missing_cut, ]
filtered_data <- filtered_data[rowSums(filtered_data == 0, na.rm = TRUE) < ncol(filtered_data) * missing_cut, ]
cat("Dimensions after filtering:", dim(filtered_data), "\n")

# 2. Optional: KNN imputation for missing values
if (apply_impute) {
  cat("Imputing missing values using KNN...\n")
  imputed_matrix <- impute.knn(as.matrix(filtered_data), k = neighbors)$data
  filtered_data <- as.data.frame(imputed_matrix)
  rownames(filtered_data) <- rownames(x)[rownames(x) %in% rownames(filtered_data)]
  cat("Missing values imputed.\n")
} else {
  cat("Skipping imputation of missing values.\n")
}

# 3. Select top variable features by standard deviation
cat("Selecting top variable features...\n")
variability <- apply(filtered_data, 1, sd, na.rm = TRUE)
sorted_indices <- order(variability, decreasing = TRUE)
filtered_data <- filtered_data[sorted_indices[1:min(top_features, nrow(filtered_data))], ]
cat("Dimensions after feature selection:", dim(filtered_data), "\n")

# 4. Optional: Min-Max normalization
if (apply_minmax) {
  cat("Applying Min-Max normalization...\n")
  filtered_data <- minmax_normalize(filtered_data)
  cat("Dimensions after normalization:", dim(filtered_data), "\n")
} else {
  cat("Skipping Min-Max normalization.\n")
}

# Save results
write.csv(filtered_data, output_file, row.names = TRUE)
cat("Processed data saved to:", output_file, "\n")
