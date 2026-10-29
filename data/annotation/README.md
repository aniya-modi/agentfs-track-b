# Platform annotation

`gpl80_probe_to_gene_symbol.tsv` contains the `ID` and `Gene symbol` columns
from NCBI GEO's platform annotation for **GPL80**, renamed to `Probe Set ID`
and `Gene Symbol` for use with `load_probe_annotation(...)`.

GPL80 is `[Hu6800] Affymetrix Human Full Length HuGeneFL Array`, the platform
used by the Golub Hu6800 reference matrix. Its table contains 7,129 probe set
IDs. The annotation identifies 6,434 rows with a non-empty gene-symbol field;
blank and multi-symbol rows are retained so Bio-RAG can report unmapped or
ambiguous mappings instead of guessing.

**Do not use GPL91 for this matrix.** NCBI identifies GPL91 as the distinct
HG-U95A platform. Platform accessions must match the dataset's actual array.

Source: [NCBI GEO GPL80](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL80)
and the GEO annotation file:
<https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL80/annot/GPL80.annot.gz>.
The source annotation table reports build 35 and annotation date 2016-08-09.

The Golub context example loads this mapping and passes the probe IDs from the
processed matrix header to `BioRAGClient.resolve_probe(..., platform="GPL80")`.
Expression values and sample labels are not loaded or sent to NCBI/KEGG.
