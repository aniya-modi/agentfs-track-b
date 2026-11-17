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
    "planner": "Propose a small set of valid feature-selection configurations. Choose only methods in tool_catalog. Keep rationale concise.",
    "bio_refiner": "Review proposals using only the supplied aggregate metadata and resolved biological context. Preserve valid configs and explain biological relevance without making diagnostic claims.",
    "challenger": "Critically inspect the proposals for unsupported claims, unavailable tools, and unsuitable feature counts. Request revision only for a concrete issue.",
    "judge": "Select one defensible proposal. Return a valid config, concise biological_rationale, confidence from 0 to 1, and request revision only for a fixable issue.",
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
            "You are one role in a research workflow for feature-selection methods on "
            "microarray data. This workflow is for research, not clinical diagnosis. "
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
    response = requests.get(f"{host}/api/tags", timeout=timeout)
    response.raise_for_status()
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
