# Track B Month 1 Interface and Completion Notes

## Module contracts

### Metadata profiler

`profile_dataset(expression, labels, ...)` expects a numeric pandas DataFrame
with samples in rows and features in columns, plus one label per row. It returns
a JSON-safe dictionary with geometry (`n_samples`, `n_features`, `p_over_n`,
`is_high_dimensional`), data quality counts, target distribution and imbalance,
zero-value fraction, and a correlation summary over at most 256
highest-variance features by default. The returned profile does not include
expression values or sample identifiers.

### Bio-RAG

`load_probe_annotation(...)` loads the selected platform's probe-to-symbol table.
`BioRAGClient.resolve_probe(...)` uses that mapping, NCBI Gene E-utilities, and
KEGG REST. Its output is the Pydantic `ProbeGeneContext` schema. Unknown probes
and ambiguous mappings are explicit results; network errors raise `BioRAGError`
so callers can record/retry them. The client caches lookups within a process.

Do not use a platform annotation for another array platform. A probe ID alone
is not sufficient evidence to identify a gene. Record the annotation source
and platform in the caller's dataset metadata.

### Selection configuration and agent state

`FeatureSelectionConfig` allows only the methods `hypergraph_centrality`,
`mrmr`, and `lasso`; requires integer `k` from 5 through 500; and constrains
`knockoff_fdr` to 0 through 0.10. Extra keys are rejected. The LLM-facing
contract contains structured fields, never generated Python.

`AgentDebateDecision` requires a validated config, biological rationale, and
confidence in [0, 1]. `AgentState` carries profiles, probe IDs, context records,
proposed configs, final decision, and errors; expression values stay in the
local data pipeline.

## Month 1 completion checklist

- [x] Python package metadata and environment instructions
- [x] Metadata profiler module and JSON example
- [x] NCBI/KEGG client with platform annotation input and provenance fields
- [x] Pydantic v2 execution/configuration schemas
- [x] Typed orchestration state and Track A/C handoff contracts
- [ ] Select the exact benchmark dataset and array platform with the team
- [ ] Obtain and record the matching platform annotation source/columns
- [ ] Configure `NCBI_EMAIL` locally and run a live lookup in the project environment
- [ ] Confirm Track A/C agree to the documented field names and types

The last four items require team or local environment choices and live API
credentials, so the code intentionally does not guess them.
