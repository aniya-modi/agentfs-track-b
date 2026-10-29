# Reference dataset inputs

## Golub leukemia / Hu6800

`golub_hu6800.csv.gz` is a small, compressed reference export of the
`Golub_Merge` ExpressionSet from Bioconductor's `golubEsets` 1.54.0 package.
It contains 72 samples, 7,129 expression features, the original Hu6800 probe
set IDs as feature headers, and the `ALL.AML` labels renamed to `target`.
Rows are samples and feature columns retain the feature order in the package.

This checked-in export makes the IDs and their order travel with the project.
The upstream scikit-feature `ALLAML.mat` file contains only `X` and `Y`; it
does not contain feature names, so it is not used for Golub ingestion.

To recreate the export, install Bioconductor `golubEsets` 1.54.0 and its
`Biobase` dependency, then run `Rscript scripts/export_golub_reference.R`:

```sh
Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager"); BiocManager::install("golubEsets")'
Rscript scripts/export_golub_reference.R
```

Sources:

- Bioconductor package: <https://bioconductor.org/packages/release/data/experiment/html/golubEsets.html>
- Package object documentation: <https://bioconductor.org/packages/release/data/experiment/manuals/golubEsets/man/golubEsets.pdf>
- Original study: Golub et al., *Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring*, Science (1999).

The Colon `.mat` source does not contain feature names. The notebook gives its
columns stable `att_1` through `att_2000` source-position identifiers. These
are dataset feature indices, not verified microarray probe IDs or gene IDs;
Bio-RAG must not resolve them without a matching platform annotation.
