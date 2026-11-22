"""Structured-output runners for a local Ollama chat model."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
from pydantic import BaseModel

from .agent_graph import AgentRunners
from .schemas import BioRefinerOutput, ChallengerOutput, JudgeOutput, PlannerOutput


class OllamaRunnerError(RuntimeError):
    """Raised when Ollama is unreachable or returns an invalid structured result."""


_ROLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "planner": PlannerOutput,
    "bio_refiner": BioRefinerOutput,
    "challenger": ChallengerOutput,
    "judge": JudgeOutput,
}

_ROLE_INSTRUCTIONS = {
    "planner": "Propose up to three distinct available selector configurations with k between 5 and 500. Choose only methods listed in tool_catalog. The local selector tools will execute them; do not claim features have been selected yet.",
    "bio_refiner": "Review the measured candidate feature sets and the supplied resolved probe-to-gene/pathway records. Prefer candidates with real pathway support only when the records show it. List pathway_supported_feature_ids only from supplied resolved contexts that contain pathways. Do not invent biology or gene functions.",
    "challenger": "Challenge the measured candidate results: inspect nested-CV balanced accuracy and macro-F1, split stability, pairwise correlation, and pathway coverage. Flag weak or unsupported evidence. Request revision only for a concrete concern.",
    "judge": "Choose only a configuration present in selection_candidates. Prioritize nested-CV balanced accuracy, then macro-F1, stability, and pathway support; use lower redundancy as a tie-breaker. Do not claim diagnosis or accuracy improvement beyond the supplied metrics.",
}


class OllamaStructuredRunner:
    """Call Ollama's local ``/api/chat`` endpoint and validate its JSON output."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout

    def __call__(self, context: dict[str, Any]) -> BaseModel:
        role = context.get("role")
        schema = _ROLE_SCHEMAS.get(role)
        if schema is None:
            raise OllamaRunnerError(f"Unsupported agent role: {role!r}")

        system_prompt = (
            "You are one role in a local research workflow for supervised feature "
            "selection on numeric tabular data. The expression matrix remains in "
            "local Python selector tools; you receive only selected IDs and summary "
            "evidence. This workflow is for research, not clinical diagnosis. "
            "Use only the supplied context and available tools. Do not invent pathway "
            "evidence. Return only data matching the required JSON schema. "
            + _ROLE_INSTRUCTIONS[role]
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            "format": schema.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 900},
            "keep_alive": "10m",
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return schema.model_validate_json(content)
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise OllamaRunnerError(f"Ollama {role} call failed: {exc}") from exc


def check_ollama_model(
    *, model: str | None = None, base_url: str | None = None, timeout: float = 5.0
) -> bool:
    """Return whether the requested model is available in the local Ollama service."""
    selected_model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    host = (base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return False
    return any(
        item.get("name", "").split(":", 1)[0] == selected_model.split(":", 1)[0]
        for item in response.json().get("models", [])
    )


def create_ollama_runners(
    *, model: str | None = None, base_url: str | None = None, timeout: float = 180.0
) -> AgentRunners:
    """Create all four LangGraph role runners backed by one local Ollama model."""
    runner = OllamaStructuredRunner(model=model, base_url=base_url, timeout=timeout)
    return AgentRunners(
        planner=runner,
        bio_refiner=runner,
        challenger=runner,
        judge=runner,
    )
