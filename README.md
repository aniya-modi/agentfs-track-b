# AgentFS Track B

Track B provides the metadata and biological context layer for the
pathway-informed feature-selection project. Month 1 delivers a local metadata
profiler, a platform-annotation-based NCBI/KEGG client, and validated contracts
for the later agent workflow.

## Environment setup

Use Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[agents]"
```

The project uses Pydantic v2, Biopython, Requests, pandas, and NumPy. The
`agents` extra installs LangGraph for the later orchestration phase. Keep API
credentials in environment variables; never commit tokens or `.env` files.

## Dataset feature identifiers

The data-loading notebook reads Golub leukemia from the checked-in
`data/reference/golub_hu6800.csv.gz` export. Its feature headers are the
original Hu6800 probe set IDs and remain attached during variance filtering.
The scikit-feature Golub `.mat` file contains no feature names, so it is not
used for this dataset. See `data/reference/README.md` for provenance.

The Colon `.mat` file also contains no feature names. Its columns receive
stable `att_1` through `att_2000` source-position labels. These are not gene or
probe IDs and must not be sent to Bio-RAG unless a verified annotation maps
them to a biological identifier.

NCBI E-utilities requires an email address. An API key is optional:

```bash
export NCBI_EMAIL="your-email@example.com"
export NCBI_API_KEY="your-ncbi-api-key"  # optional
```

KEGG REST access does not use an API key in this client. Its request rate is
throttled, and transient server/network errors are retried.

## Metadata profiler

Rows are samples; columns are numeric features. The result is aggregate-only
JSON metadata and does not contain expression values or sample identifiers:

```python
import pandas as pd

from agentfs.metadata_profiler import profile_dataset, profile_to_json

expression = pd.DataFrame({"probe_a": [1.0, 2.0, 3.0], "probe_b": [5.0, 5.0, 5.0]})
labels = pd.Series(["ALL", "ALL", "AML"])
profile = profile_dataset(expression, labels, dataset_id="example")
print(profile_to_json(profile))
```

`examples/profile_example.py` provides a small runnable example. The profile
includes sample and feature counts, p/n ratio, missing and non-finite counts,
near-zero-variance count, zero-value fraction, target class counts, imbalance
ratio, and a bounded correlation summary. If profile statistics will influence
cross-validation settings, calculate them on the current training fold.

## Bio-RAG probe resolution

Opaque probe IDs require the correct platform annotation file. This client does
not guess gene symbols from probe IDs. Golub uses the Hu6800 array, which NCBI
identifies as GPL80; its checked-in annotation is in
`data/annotation/gpl80_probe_to_gene_symbol.tsv`. GPL91 is the different
HG-U95A platform and must not be used for the Golub matrix.

Load that mapping and resolve a probe through it before NCBI/KEGG lookup:

```python
from agentfs.bio_rag import BioRAGClient, load_probe_annotation

annotation = load_probe_annotation(
    "data/annotation/gpl80_probe_to_gene_symbol.tsv",
    probe_column="Probe Set ID",
    gene_symbol_column="Gene Symbol",
)
client = BioRAGClient(
    annotation,
    organism="Homo sapiens",
    kegg_organism="hsa",
    annotation_source="NCBI GEO GPL80 annotation",
)
context = client.resolve_probe("M25079_s_at", platform="GPL80")
print(context.model_dump_json(indent=2))
```

The annotation file retains blank and multi-symbol rows. The client reports
unmapped or ambiguous probe mappings explicitly instead of choosing a symbol
without evidence. See `examples/resolve_golub_context.py` for a bounded live
lookup that reads only the processed CSV header. Set `NCBI_EMAIL` before running
it. Only mapped gene symbols and identifiers go to NCBI/KEGG; expression values
and sample labels remain local. Annotation provenance and platform are included
in each context result.

Run a small live lookup from the repository root:

```bash
export NCBI_EMAIL="your-email@example.com"
python examples/resolve_golub_context.py --limit 5
```

## Track handoff contracts

- Track A receives `metadata_profiler.profile_dataset(...)` output and a list of
  `ProbeGeneContext` records containing gene and pathway identifiers.
- Track C receives `FeatureSelectionConfig` and `AgentDebateDecision` schemas.
- `AgentState` carries metadata, identifiers, validated proposals, and errors;
  it does not include expression values.

See [`docs/track_b_month1.md`](docs/track_b_month1.md) for the Month 1 contract
and completion checklist.
