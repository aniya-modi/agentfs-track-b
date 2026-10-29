if (!requireNamespace("golubEsets", quietly = TRUE)) {
  stop('Install the Bioconductor package with BiocManager::install("golubEsets").')
}
if (!requireNamespace("Biobase", quietly = TRUE)) {
  stop('Install the Bioconductor dependency with BiocManager::install("Biobase").')
}

package_version <- as.character(utils::packageVersion("golubEsets"))
if (package_version != "1.54.0") {
  stop("Expected golubEsets 1.54.0; found ", package_version, ".")
}

data("Golub_Merge", package = "golubEsets", envir = environment())
expression <- t(Biobase::exprs(Golub_Merge))
colnames(expression) <- Biobase::featureNames(Golub_Merge)
dataset <- as.data.frame(expression, check.names = FALSE)
dataset$target <- as.character(Biobase::pData(Golub_Merge)[["ALL.AML"]])

output_path <- "data/reference/golub_hu6800.csv.gz"
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
output <- gzfile(output_path, open = "wt")
tryCatch(
  utils::write.csv(dataset, output, row.names = FALSE, quote = FALSE),
  finally = close(output)
)
message("Wrote ", output_path, " from golubEsets ", package_version)
