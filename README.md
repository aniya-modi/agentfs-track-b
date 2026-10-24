# AgentFS Track B

Track B implements dataset metadata profiling, biological context retrieval, and
validated configuration for the pathway-informed feature-selection project.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[agents,dev]"
```

Run the synthetic profiler example:

```bash
python examples/profile_example.py
```

For NCBI Entrez lookups, set an email address and optionally an API key:

```bash
export NCBI_EMAIL="your-email@example.com"
export NCBI_API_KEY="your-ncbi-api-key"  # optional
```

Do not commit API keys, patient-level data, or raw datasets. Bio-RAG currently
accepts gene symbols; probe-ID mapping depends on the final dataset platform and
annotation source, which should be agreed with the project team.

## Modules

- `metadata_profiler.py`: aggregate dimensions, missingness, variance, and class balance.
- `bio_rag.py`: retrieve NCBI Gene summaries for gene symbols.
- `schemas.py`: validate selection settings and final agent decisions.
- `agent_state.py`: initial shared state shape for later orchestration.
