"""Run local feature selection, Bio-RAG review, and the four-agent decision graph.

Input format: a CSV with one sample per row, numeric feature columns, and one
categorical or numeric target column. The expression/feature matrix and labels
remain in local Python selector tools and are never sent to Ollama.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from agentfs.agent_graph import build_agent_graph
from agentfs.bio_rag import BioRAGClient, BioRAGError, load_probe_annotation
from agentfs.kegg import build_kegg_incidence
from agentfs.metadata_profiler import profile_dataset
from agentfs.ollama_runners import check_ollama_model, create_ollama_runners
from agentfs.selection_pipeline import make_local_selection_executor
from agentfs.tool_registry import FeatureSelectionToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/reference/golub_hu6800.csv.gz"
DEFAULT_ANNOTATION = PROJECT_ROOT / "data/annotation/gpl80_probe_to_gene_symbol.tsv"
DEFAULT_ANNOTATION_SOURCE = "NCBI GEO GPL80 annotation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--label-column", default="target")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help=(
            "Optional platform-matched probe-to-symbol TSV/CSV. The default Golub "
            "dataset automatically uses the checked-in GPL80 annotation."
        ),
    )
    parser.add_argument("--platform", default=None, help="Annotation platform, such as GPL80.")
    parser.add_argument("--probe-column", default="Probe Set ID")
    parser.add_argument("--gene-symbol-column", default="Gene Symbol")
    parser.add_argument("--annotation-source", default=None)
    parser.add_argument("--organism", default="Homo sapiens")
    parser.add_argument("--kegg-organism", default="hsa")
    parser.add_argument(
        "--bio-rag-limit",
        type=int,
        default=12,
        help="Maximum selected feature IDs to resolve through live NCBI/KEGG; 0 disables it.",
    )
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--inner-splits", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=3)
    args = parser.parse_args()
    if args.bio_rag_limit < 0 or args.max_candidates < 1:
        parser.error("--bio-rag-limit must be non-negative and --max-candidates positive")
    return args


def _load_dataset(path: Path, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if not path.is_file():
        raise SystemExit(f"Dataset file not found: {path}")
    frame = pd.read_csv(path)
    if label_column not in frame:
        raise SystemExit(f"Target column {label_column!r} was not found in {path}.")
    labels = frame[label_column]
    if labels.isna().any():
        raise SystemExit(f"Target column {label_column!r} contains missing values.")
    if labels.nunique(dropna=True) < 2:
        raise SystemExit(f"Target column {label_column!r} must contain at least two classes.")
    expression = frame.drop(columns=[label_column])
    if expression.empty:
        raise SystemExit("The dataset has no feature columns.")
    nonnumeric = [
        column
        for column in expression.columns
        if not pd.api.types.is_numeric_dtype(expression[column].dtype)
    ]
    if nonnumeric:
        raise SystemExit(
            "This demo currently expects numeric feature columns. Encode or remove "
            f"categorical features first. Non-numeric columns: {nonnumeric[:10]}"
        )
    if not expression.columns.is_unique:
        raise SystemExit("Feature identifiers must be unique.")
    return expression, labels


def _contexts_from_biorag(
    *,
    annotation: dict[str, list[str]],
    annotation_source: str,
    organism: str,
    kegg_organism: str,
    platform: str | None,
    limit: int,
):
    if limit == 0:
        print("Bio-RAG live lookups disabled (--bio-rag-limit 0).")
        return None
    if not os.environ.get("NCBI_EMAIL"):
        print("NCBI_EMAIL is unset; skipping live Bio-RAG. KEGG hypergraph selection remains enabled.")
        return None

    client = BioRAGClient(
        annotation,
        organism=organism,
        kegg_organism=kegg_organism,
        annotation_source=annotation_source,
    )

    def resolve(feature_ids: list[str]):
        contexts = []
        for feature_id in feature_ids[:limit]:
            try:
                contexts.append(client.resolve_probe(feature_id, platform=platform))
            except BioRAGError as exc:
                print(f"Bio-RAG lookup failed for {feature_id}: {exc}")
        return contexts

    return resolve


def _print_role_outputs(result: dict[str, Any]) -> None:
    for role, key in (
        ("Planner", "planner_output"),
        ("Local selector tools", "selection_candidates"),
        ("Bio-Refiner", "bio_refiner_output"),
        ("Challenger", "challenger_output"),
        ("Judge", "judge_output"),
    ):
        print(f"\n--- {role} ---")
        value = result.get(key)
        if value is None:
            print("No output recorded.")
        elif isinstance(value, list):
            for item in value:
                print(item.model_dump_json(indent=2))
        else:
            print(value.model_dump_json(indent=2))

    decision = result.get("final_decision")
    print("\n--- Final selected feature set ---")
    if decision is None:
        print("No final decision was produced.")
        return
    print(f"Method: {decision.config.method}; k: {decision.config.k}")
    print(f"Features selected: {len(decision.selected_feature_ids)}")
    print(f"Nested-CV accuracy: {decision.cv_accuracy}")
    print(f"Nested-CV balanced accuracy: {decision.cv_balanced_accuracy}")
    print(f"Nested-CV macro F1: {decision.cv_macro_f1}")
    print(f"Split stability (mean Jaccard): {decision.selection_stability}")
    print("Selected feature IDs:")
    for feature_id in decision.selected_feature_ids:
        print(f"- {feature_id}")
    print(f"Rationale: {decision.biological_rationale}")


def main() -> None:
    args = _parse_args()
    if not check_ollama_model():
        raise SystemExit(
            "Ollama is not running or Llama 3.2 is missing. Start Ollama and run "
            "`ollama pull llama3.2`."
        )

    expression, labels = _load_dataset(args.dataset, args.label_column)
    dataset_id = args.dataset_id or args.dataset.stem.removesuffix(".csv")
    profile = profile_dataset(expression, labels, dataset_id=dataset_id)
    feature_ids = [str(item) for item in expression.columns]
    print(
        f"Loaded {dataset_id}: {len(expression)} samples × {len(feature_ids)} numeric features; "
        f"classes={profile['class_counts']}"
    )

    annotation_path = args.annotation
    platform = args.platform
    if annotation_path is None and args.dataset.resolve() == DEFAULT_DATASET.resolve():
        annotation_path = DEFAULT_ANNOTATION
        platform = platform or "GPL80"

    annotation: dict[str, list[str]] = {}
    annotation_source = args.annotation_source or "user-supplied platform annotation"
    registry = FeatureSelectionToolRegistry()
    context_resolver = None
    if annotation_path is not None:
        annotation = load_probe_annotation(
            annotation_path,
            probe_column=args.probe_column,
            gene_symbol_column=args.gene_symbol_column,
        )
        annotation_source = args.annotation_source or (
            DEFAULT_ANNOTATION_SOURCE if annotation_path == DEFAULT_ANNOTATION else str(annotation_path)
        )
        incidence, hashes = build_kegg_incidence(
            feature_ids,
            annotation_path=annotation_path,
            probe_column=args.probe_column,
            gene_symbol_column=args.gene_symbol_column,
            kegg_organism=args.kegg_organism,
        )
        registry = FeatureSelectionToolRegistry(H=incidence)
        print(f"KEGG incidence: {incidence.shape[0]} features × {incidence.shape[1]} pathways; {incidence.nnz} memberships")
        for filename, digest in hashes.items():
            print(f"KEGG snapshot SHA-256 ({filename}): {digest}")
        context_resolver = _contexts_from_biorag(
            annotation=annotation,
            annotation_source=annotation_source,
            organism=args.organism,
            kegg_organism=args.kegg_organism,
            platform=platform,
            limit=args.bio_rag_limit,
        )
    else:
        print("No platform annotation supplied; hypergraph and Bio-RAG tools are unavailable.")

    executor = make_local_selection_executor(
        expression,
        labels,
        registry=registry,
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        inner_splits=args.inner_splits,
        max_candidates=args.max_candidates,
    )
    graph = build_agent_graph(
        runners=create_ollama_runners(),
        registry=registry,
        selection_executor=executor,
        context_resolver=context_resolver,
    )
    result = graph.invoke(
        {
            "dataset_id": dataset_id,
            "metadata_profile": profile,
            # The full feature list stays inside the selector closure; only a
            # small identifier sample is needed in graph state and prompts.
            "probe_ids": feature_ids[:20],
            "gene_contexts": [],
            "max_agent_retries": 1,
            "max_revision_rounds": 1,
        }
    )
    print(f"\nCV settings: outer={args.cv_splits} folds × {args.cv_repeats} repeat; inner={args.inner_splits} folds")
    print(
        "These candidate CV scores are exploratory, used for selector choice, "
        "and are not an unbiased final estimate or clinical evidence."
    )
    _print_role_outputs(result)
    print(f"\nGraph status: {result['status']}; fallback used: {result.get('fallback_used', False)}")
    if result.get("errors"):
        print("Workflow notes:")
        for error in result["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
