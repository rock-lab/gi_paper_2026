#!/usr/bin/env Rscript

# GAM Correction for Genetic Interaction Scores
# Usage: Rscript gam_correction_standalone.R input.tsv output.tsv [sp] [k] [method]
#
# Arguments:
#   input.tsv: TSV file with columns: guide_pair, expected, y25
#   output.tsv: Output TSV file with corrected values
#   sp: Smoothing parameter (optional, "NULL" for automatic)
#   k: Basis dimension (optional, default: 20)
#   method: Fitting method (optional, default: "ML")

suppressPackageStartupMessages({
  library(mgcv)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript gam_correction_standalone.R input.tsv output.tsv [sp] [k] [method]")
}

input_file <- args[1]
output_file <- args[2]
sp_param <- if (length(args) >= 3 && args[3] != "NULL") as.numeric(args[3]) else NULL
k_param <- if (length(args) >= 4) as.integer(args[4]) else 20
method_param <- if (length(args) >= 5) args[5] else "ML"

# Read input data
cat("Reading input data from:", input_file, "\n")
data <- read.table(input_file, header=TRUE, sep="\t", stringsAsFactors=FALSE)

# Check required columns
required_cols <- c("guide_pair", "expected", "y25")
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse=", ")))
}

# Calculate Y25_delta (uncorrected GI score)
data$y25_delta <- data$y25 - data$expected

# Remove rows with NA values
data_clean <- data[complete.cases(data[c("expected", "y25_delta")]), ]
cat("Fitting GAM on", nrow(data_clean), "guide pairs\n")

# Fit GAM model: y25_delta ~ s(expected)
# The GAM models the relationship between expected fitness and deviation from expectation
if (!is.null(sp_param)) {
  # Use specified smoothing parameter
  cat("Fitting GAM with sp =", sp_param, "and k =", k_param, "\n")
  gam_model <- gam(y25_delta ~ s(expected, sp=sp_param, k=k_param),
                   data=data_clean, method=method_param)
} else {
  # Let GAM choose smoothing parameter automatically
  cat("Fitting GAM with automatic smoothing parameter selection (k =", k_param, ")\n")
  gam_model <- gam(y25_delta ~ s(expected, k=k_param),
                   data=data_clean, method=method_param)
}

# Print model summary
cat("\nGAM Model Summary:\n")
cat("R-squared (adjusted):", summary(gam_model)$r.sq, "\n")
cat("Deviance explained:", summary(gam_model)$dev.expl, "\n")
cat("Estimated degrees of freedom:", sum(gam_model$edf), "\n")

# Get predictions (systematic bias to remove)
data_clean$gam_prediction <- predict(gam_model, newdata=data_clean)

# Calculate corrected Y25_delta (residuals from GAM)
data_clean$y_corrected <- data_clean$y25_delta - data_clean$gam_prediction

# Merge back with original data to maintain all guide pairs
result <- merge(data[c("guide_pair")],
                data_clean[c("guide_pair", "y_corrected", "gam_prediction")],
                by="guide_pair", all.x=TRUE)

# Write output
cat("\nWriting corrected values to:", output_file, "\n")
write.table(result, file=output_file, sep="\t", row.names=FALSE, quote=FALSE)

cat("GAM correction completed successfully\n")
cat("Mean uncorrected Y25_delta:", mean(data_clean$y25_delta, na.rm=TRUE), "\n")
cat("Mean corrected Y25_delta:", mean(data_clean$y_corrected, na.rm=TRUE), "\n")
cat("SD uncorrected Y25_delta:", sd(data_clean$y25_delta, na.rm=TRUE), "\n")
cat("SD corrected Y25_delta:", sd(data_clean$y_corrected, na.rm=TRUE), "\n")