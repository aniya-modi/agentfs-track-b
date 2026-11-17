"""Run the four-agent research workflow with a local Ollama model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agentfs.agent_graph import build_agent_graph
from agentfs.metadata_profiler import profile_dataset
from agentfs.ollama_runners import check_ollama_model, create_ollama_runners

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLUB_PATH = PROJECT_ROOT / "data/reference/golub_hu6800.csv.gz"


def main() -> None:
    if not check_ollama_model():
        raise SystemExit(
            "Ollama is not running or Llama 3.2 is missing. Start Ollama and run "
            "`ollama pull llama3.2`."
        )

    frame = pd.read_csv(GOLUB_PATH, compression="gzip")
    labels = (frame["target"].astype(str) == "AML").astype(int)
    expression = frame.drop(columns=["target"])
    profile = profile_dataset(expression, labels, dataset_id="golub-hu6800")

    graph = build_agent_graph(runners=create_ollama_runners())
    result = graph.invoke(
        {
            "dataset_id": "golub-hu6800",
            "metadata_profile": profile,
            "probe_ids": list(expression.columns),
            "gene_contexts": [],
            "max_agent_retries": 1,
            "max_revision_rounds": 1,
        }
    )

    decision = result.get("final_decision")
    print(f"Status: {result['status']}")
    print(f"Fallback used: {result.get('fallback_used', False)}")
    print(f"Agent attempts: {result.get('agent_attempts', {})}")
    if decision:
        print(f"Selected configuration: {decision.config.model_dump()}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Rationale: {decision.biological_rationale}")
    if result.get("errors"):
        print("Recoverable agent errors:")
        for error in result["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
