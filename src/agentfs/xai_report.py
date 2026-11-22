"""Evidence-grounded natural-language reports for retained biomarkers.

The generator is deterministic: every sentence is derived from supplied
selection metadata, platform annotation, NCBI/KEGG context, signed effects,
and optional enrichment terms. It deliberately adds no external literature and
makes no clinical claims.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .schemas import ProbeGeneContext, StrictSchema

MAX_SUMMARY_CHARS = 400
MAX_PATHWAYS_IN_NARRATIVE = 2

EvidenceKind = Literal[
    "retention",
    "effect",
    "annotation",
    "pathway",
    "gene_summary",
    "enrichment",
    "limitation",
]
EvidenceStrength = Literal["strong", "moderate", "limited"]
EvidenceLevel = Literal["high", "moderate", "low"]


class RetentionEvidence(StrictSchema):
    """Upstream feature-selection evidence for one retained probe."""

    method: str = Field(min_length=1, max_length=50)
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None
    requested_k: int | None = Field(default=None, ge=1)


class BiomarkerEvidence(StrictSchema):
    """One retained biomarker and the evidence supplied for explaining it."""

    context: ProbeGeneContext
    retention: RetentionEvidence | None = None
    signed_effect_size: float | None = None
    effect_metric: str = Field(default="signed effect size", min_length=1, max_length=50)


class EnrichmentTerm(StrictSchema):
    """One supplied enrichment result, such as a preranked GSEA term."""

    term: str = Field(min_length=1, max_length=300)
    nes: float | None = None
    fdr: float | None = Field(default=None, ge=0.0, le=1.0)
    direction: str | None = Field(default=None, max_length=100)
    lead_genes: list[str] = Field(default_factory=list)
    source: str = Field(default="supplied enrichment summary", max_length=100)


class ExplanationReason(StrictSchema):
    """One traceable piece of evidence used in a biomarker explanation."""

    kind: EvidenceKind
    statement: str = Field(min_length=1)
    source: str | None = None
    strength: EvidenceStrength = "moderate"


class BiomarkerExplanation(StrictSchema):
    """Structured explanation and rendered narrative for one biomarker."""

    probe_id: str
    gene_symbol: str | None = None
    status: str
    direction: str
    signed_effect_size: float | None = None
    retention_method: str | None = None
    retention_rank: int | None = None
    evidence_level: EvidenceLevel
    narrative: str
    reasons: list[ExplanationReason]
    citations: list[str]


class XAIReport(StrictSchema):
    """Complete report that can be serialized to JSON or Markdown."""

    title: str
    dataset_id: str
    positive_class: str
    negative_class: str
    generated_at: datetime
    scope: str
    summary: str
    pathway_themes: list[str]
    enrichment_summary: str | None = None
    enrichment_terms: list[EnrichmentTerm] = Field(default_factory=list)
    biomarkers: list[BiomarkerExplanation] = Field(min_length=1)
    limitations: list[str]

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"_Generated {self.generated_at.isoformat()}._",
            "",
            "## Scope",
            "",
            self.scope,
            "",
            "## Summary",
            "",
            self.summary,
        ]
        if self.enrichment_summary:
            lines.extend(["", self.enrichment_summary])
        lines.extend(["", "### Pathway themes", ""])
        if self.pathway_themes:
            lines.extend(f"- {theme}" for theme in self.pathway_themes)
        else:
            lines.append("- No pathway theme recurred across the supplied biomarkers.")

        if self.enrichment_terms:
            lines.extend(
                [
                    "",
                    "### Supplied enrichment evidence",
                    "",
                    "| Term | NES | FDR | Direction |",
                    "| --- | ---: | ---: | --- |",
                ]
            )
            for term in self.enrichment_terms[:5]:
                nes = f"{term.nes:.2f}" if term.nes is not None else "n/a"
                fdr = _format_fdr_value(term.fdr)
                direction = _readable_direction(
                    term.direction, self.positive_class, self.negative_class
                )
                lines.append(f"| {term.term} | {nes} | {fdr} | {direction} |")

        lines.extend(["", "## Biomarker explanations", ""])
        for index, biomarker in enumerate(self.biomarkers, start=1):
            label = biomarker.gene_symbol or biomarker.probe_id
            lines.extend([f"### {index}. {label} (`{biomarker.probe_id}`)", ""])
            retention = (
                f"{biomarker.retention_method} rank {biomarker.retention_rank}"
                if biomarker.retention_method and biomarker.retention_rank
                else biomarker.retention_method or "not supplied"
            )
            effect = (
                f"{biomarker.signed_effect_size:+.3f}"
                if biomarker.signed_effect_size is not None
                else "n/a"
            )
            lines.append(
                f"**Retention:** {retention} | **Direction:** {biomarker.direction} | "
                f"**Effect:** {effect} | **Evidence completeness:** {biomarker.evidence_level}"
            )
            lines.extend(["", biomarker.narrative, "", "**Evidence**"])
            for reason in biomarker.reasons:
                source = f" _(Source: {reason.source})_" if reason.source else ""
                kind = reason.kind.replace("_", " ").title()
                lines.append(f"- **{kind}** ({reason.strength}): {reason.statement}{source}")
            if biomarker.citations:
                lines.extend(["", f"**Citations:** {'; '.join(biomarker.citations)}"])
            lines.append("")

        lines.extend(["## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines).rstrip() + "\n"


def _symbol(context: ProbeGeneContext) -> str | None:
    if context.gene_symbol:
        return context.gene_symbol
    if len(context.gene_symbols) == 1:
        return context.gene_symbols[0]
    return None


def _clean_pathway_name(name: str) -> str:
    for suffix in (" - Homo sapiens (human)", " - Homo sapiens", " (human)"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _direction(effect: float | None, positive_class: str, negative_class: str) -> str:
    if effect is None:
        return "not supplied"
    if effect > 0:
        return f"higher in {positive_class}"
    if effect < 0:
        return f"higher in {negative_class}"
    return "no class difference"


_METHOD_LABELS = {
    "mrmr": "mRMR",
    "lasso": "LASSO",
    "hypergraph_centrality": "hypergraph centrality",
}


def _method_label(method: str) -> str:
    return _METHOD_LABELS.get(method.casefold(), method)


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _format_fdr_value(fdr: float | None) -> str:
    if fdr is None:
        return "n/a"
    if fdr < 0.001:
        return "<0.001"
    return f"{fdr:.3g}"


def _fdr_label(fdr: float | None) -> str:
    if fdr is None:
        return "FDR not supplied"
    if fdr < 0.001:
        return "FDR<0.001"
    return f"FDR={fdr:.3g}"


def _readable_direction(
    value: str | None, positive_class: str, negative_class: str
) -> str:
    if not value:
        return "n/a"
    if value in {"up_in_positive_class", "higher_in_positive_class", "positive"}:
        return f"higher in {positive_class}"
    if value in {"down_in_positive_class", "higher_in_negative_class", "negative"}:
        return f"higher in {negative_class}"
    return value.replace("_", " ")


def _term_direction(
    term: EnrichmentTerm, positive_class: str, negative_class: str
) -> str | None:
    readable = _readable_direction(term.direction, positive_class, negative_class)
    if readable != "n/a":
        return readable
    if term.nes is not None and term.nes != 0:
        return f"higher in {positive_class if term.nes > 0 else negative_class}"
    return None


def _direction_sign(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def _term_sign(term: EnrichmentTerm) -> int:
    if term.nes is not None:
        return _direction_sign(term.nes)
    if term.direction in {"up_in_positive_class", "higher_in_positive_class", "positive"}:
        return 1
    if term.direction in {"down_in_positive_class", "higher_in_negative_class", "negative"}:
        return -1
    return 0


def _concordance_note(
    term: EnrichmentTerm,
    effect: float | None,
    positive_class: str,
    negative_class: str,
) -> str:
    gene_sign = _direction_sign(effect)
    term_sign = _term_sign(term)
    if gene_sign == 0 or term_sign == 0:
        return ""
    if gene_sign == term_sign:
        class_name = positive_class if gene_sign > 0 else negative_class
        return f" Its direction agrees with this gene being higher in {class_name}."
    return (
        " Its direction is opposite to this gene's effect, so it indicates pathway "
        "membership rather than concordant support."
    )


def _retention_clause(retention: RetentionEvidence | None) -> str:
    if retention is None:
        return "was supplied for biological explanation; no selection method or rank was provided"
    parts = [f"was retained by {_method_label(retention.method)}"]
    if retention.rank is not None:
        parts.append(f"at rank {retention.rank}")
    if retention.requested_k is not None:
        parts.append(f"(requested k={retention.requested_k})")
    if retention.score is not None:
        parts.append(f"with score {retention.score:.4g}")
    return " ".join(parts)


def _annotation_reason(context: ProbeGeneContext) -> ExplanationReason:
    symbol = _symbol(context)
    platform = context.platform or "platform"
    if context.status == "resolved" and symbol:
        gene_id = f" (NCBI Gene {context.ncbi_gene_id})" if context.ncbi_gene_id else ""
        return ExplanationReason(
            kind="annotation",
            statement=f"The {platform} probe maps unambiguously to {symbol}{gene_id}.",
            source="platform annotation",
            strength="strong",
        )
    if context.status == "ambiguous_probe_mapping":
        symbols = ", ".join(context.gene_symbols) or "multiple symbols"
        statement = (
            f"The {platform} annotation maps this probe to multiple symbols "
            f"({symbols}); gene-level biological claims are withheld."
        )
    elif context.status == "unmapped_probe":
        statement = f"The {platform} annotation did not map this probe to a gene symbol."
    elif context.status == "gene_not_found":
        statement = "The mapped gene symbol was not found in NCBI Gene."
    elif context.status == "ambiguous_gene_mapping":
        statement = "The gene symbol maps to multiple NCBI Gene records, so gene-level claims are withheld."
    else:
        statement = f"The mapping status is {context.status}; gene-level claims are withheld."
    if context.detail:
        statement = f"{statement} Detail: {context.detail}"
    return ExplanationReason(
        kind="annotation",
        statement=statement,
        source="platform annotation",
        strength="limited",
    )


def _pathway_reason(context: ProbeGeneContext) -> ExplanationReason:
    if not context.pathways:
        return ExplanationReason(
            kind="limitation",
            statement="No KEGG pathway membership was returned for this gene.",
            source="KEGG REST API",
            strength="limited",
        )
    shown = [
        f"{pathway.pathway_id} ({_clean_pathway_name(pathway.name)})"
        for pathway in context.pathways[:5]
    ]
    more = len(context.pathways) - len(shown)
    suffix = f", and {more} more" if more > 0 else ""
    return ExplanationReason(
        kind="pathway",
        statement=(
            f"Annotated to {len(context.pathways)} KEGG pathways: "
            f"{', '.join(shown)}{suffix}."
        ),
        source="KEGG REST API",
        strength="strong",
    )


def _summary_reason(context: ProbeGeneContext) -> ExplanationReason | None:
    if not context.gene_summary:
        return None
    summary = context.gene_summary.strip()
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS].rstrip() + "..."
    return ExplanationReason(
        kind="gene_summary",
        statement=f"NCBI Gene summary: \"{summary}\"",
        source="NCBI Gene",
        strength="strong",
    )


def _enrichment_matches(
    context: ProbeGeneContext,
    symbol: str | None,
    enrichment_terms: Sequence[EnrichmentTerm],
) -> list[tuple[EnrichmentTerm, str]]:
    pathway_names = [_clean_pathway_name(pathway.name).casefold() for pathway in context.pathways]
    matches: list[tuple[EnrichmentTerm, str]] = []
    for term in enrichment_terms:
        term_name = term.term.casefold()
        matched_pathway = next(
            (
                name
                for name in pathway_names
                if name in term_name or term_name in name
            ),
            None,
        )
        if matched_pathway is not None:
            matches.append((term, f"pathway annotation '{matched_pathway}'"))
        elif symbol and symbol.upper() in {gene.upper() for gene in term.lead_genes}:
            matches.append((term, "leading-edge gene list"))
    return matches


def _enrichment_reasons(
    matches: Sequence[tuple[EnrichmentTerm, str]],
    enrichment_supplied: bool,
    has_pathways: bool,
    effect: float | None,
    positive_class: str,
    negative_class: str,
) -> list[ExplanationReason]:
    reasons: list[ExplanationReason] = []
    for term, basis in matches[:2]:
        nes = f", NES={term.nes:.2f}" if term.nes is not None else ""
        fdr = f", {_fdr_label(term.fdr)}" if term.fdr is not None else ""
        term_direction = _term_direction(term, positive_class, negative_class)
        direction = f"; term direction: {term_direction}" if term_direction else ""
        note = _concordance_note(term, effect, positive_class, negative_class)
        reasons.append(
            ExplanationReason(
                kind="enrichment",
                statement=(
                    f"The supplied term '{term.term}' links to this biomarker through "
                    f"{basis}{nes}{fdr}{direction}."
                    f"{note}"
                ),
                source=term.source,
                strength="moderate",
            )
        )
    if enrichment_supplied and has_pathways and not matches:
        reasons.append(
            ExplanationReason(
                kind="limitation",
                statement="None of the supplied enrichment terms matched this biomarker's pathways or leading-edge genes.",
                source="supplied enrichment summary",
                strength="limited",
            )
        )
    return reasons


def _evidence_level(
    context: ProbeGeneContext,
    retention: RetentionEvidence | None,
    effect: float | None,
    matches: Sequence[tuple[EnrichmentTerm, str]],
) -> EvidenceLevel:
    if context.status != "resolved":
        return "low"
    evidence_count = 1
    if context.pathways:
        evidence_count += 1
    if context.gene_summary:
        evidence_count += 1
    if retention is not None:
        evidence_count += 1
    if effect is not None:
        evidence_count += 1
    if matches:
        evidence_count += 1
    if evidence_count >= 5:
        return "high"
    if evidence_count >= 3:
        return "moderate"
    return "low"


def _narrative(
    context: ProbeGeneContext,
    retention: RetentionEvidence | None,
    effect: float | None,
    effect_metric: str,
    matches: Sequence[tuple[EnrichmentTerm, str]],
    evidence_level: EvidenceLevel,
    positive_class: str,
    negative_class: str,
) -> str:
    symbol = _symbol(context)
    label = f"{symbol} ({context.probe_id})" if symbol else f"Probe {context.probe_id}"
    sentences = [f"{label} {_retention_clause(retention)}."]

    if context.status == "resolved" and symbol:
        gene_id = f" (NCBI Gene {context.ncbi_gene_id})" if context.ncbi_gene_id else ""
        sentences.append(f"The platform annotation maps it to {symbol}{gene_id}.")
    else:
        sentences.append(
            f"Its platform mapping status is {context.status}, so gene-level biological "
            "claims are withheld."
        )

    if effect is not None:
        sentences.append(
            f"The supplied {effect_metric} is {effect:+.3f}, indicating "
            f"{_direction(effect, positive_class, negative_class)}."
        )
    else:
        sentences.append("No signed effect size was supplied, so the direction is unknown.")

    if context.pathways:
        names = [_clean_pathway_name(pathway.name) for pathway in context.pathways]
        shown = ", ".join(names[:MAX_PATHWAYS_IN_NARRATIVE])
        sentences.append(
            f"It is annotated to {len(context.pathways)} KEGG pathways, including {shown}."
        )
    elif context.status == "resolved":
        sentences.append("No KEGG pathway membership was returned for this gene.")

    if matches:
        term, basis = matches[0]
        detail = f"NES={term.nes:.2f}" if term.nes is not None else "NES not supplied"
        fdr = _fdr_label(term.fdr)
        term_direction = _term_direction(term, positive_class, negative_class)
        direction = f", term direction: {term_direction}" if term_direction else ""
        note = _concordance_note(term, effect, positive_class, negative_class)
        sentences.append(
            f"The supplied enrichment term '{term.term}' links to this biomarker through "
            f"its {basis} ({detail}, {fdr}{direction}).{note}"
        )

    if context.gene_summary:
        first_sentence = context.gene_summary.split(". ")[0].strip()
        if first_sentence:
            sentences.append(f"The NCBI Gene summary begins: \"{first_sentence}.\"")

    sentences.append(
        f"Evidence completeness is {evidence_level} based on the supplied "
        "annotation, pathway, selection, and effect fields."
    )
    sentences.append("This is a research explanation, not a clinical interpretation.")
    return " ".join(sentences)


def generate_xai_report(
    dataset_id: str,
    biomarkers: Sequence[BiomarkerEvidence | Mapping[str, Any]],
    *,
    positive_class: str = "positive class",
    negative_class: str = "negative class",
    enrichment_terms: Sequence[EnrichmentTerm | Mapping[str, Any]] = (),
    generated_at: datetime | None = None,
) -> XAIReport:
    """Generate a deterministic XAI report from supplied biomarker evidence."""
    evidence_items = [
        item if isinstance(item, BiomarkerEvidence) else BiomarkerEvidence.model_validate(item)
        for item in biomarkers
    ]
    if not evidence_items:
        raise ValueError("At least one biomarker is required.")
    terms = [
        term if isinstance(term, EnrichmentTerm) else EnrichmentTerm.model_validate(term)
        for term in enrichment_terms
    ]

    explanations: list[BiomarkerExplanation] = []
    theme_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    matched_terms: set[str] = set()
    resolved_count = 0
    pathway_count = 0

    for item in evidence_items:
        context = item.context
        symbol = _symbol(context)
        direction = _direction(item.signed_effect_size, positive_class, negative_class)
        if item.signed_effect_size is not None:
            direction_counts[direction] += 1
        if context.status == "resolved":
            resolved_count += 1
        if context.pathways:
            pathway_count += 1
            for pathway in context.pathways:
                theme_counts[_clean_pathway_name(pathway.name)] += 1

        matches = _enrichment_matches(context, symbol, terms)
        matched_terms.update(term.term for term, _ in matches)
        level = _evidence_level(
            context, item.retention, item.signed_effect_size, matches
        )

        reasons: list[ExplanationReason] = []
        if item.retention is not None:
            retention = item.retention
            detail = _method_label(retention.method)
            if retention.rank is not None:
                detail += f" rank {retention.rank}"
            if retention.requested_k is not None:
                detail += f" (requested k={retention.requested_k})"
            if retention.score is not None:
                detail += f" with score {retention.score:.4g}"
            reasons.append(
                ExplanationReason(
                    kind="retention",
                    statement=f"Retained by {detail}.",
                    source="supplied selection metadata",
                    strength="strong",
                )
            )
        else:
            reasons.append(
                ExplanationReason(
                    kind="limitation",
                    statement="Selection method and rank were not supplied, so retention cannot be traced to a selector.",
                    strength="limited",
                )
            )
        if item.signed_effect_size is not None:
            reasons.append(
                ExplanationReason(
                    kind="effect",
                    statement=(
                        f"{_sentence_case(item.effect_metric)} is "
                        f"{item.signed_effect_size:+.3f} ({direction})."
                    ),
                    source="supplied effect-size metadata",
                    strength="moderate",
                )
            )
        reasons.append(_annotation_reason(context))
        reasons.append(_pathway_reason(context))
        summary_reason = _summary_reason(context)
        if summary_reason is not None:
            reasons.append(summary_reason)
        reasons.extend(
            _enrichment_reasons(
                matches,
                bool(terms),
                bool(context.pathways),
                item.signed_effect_size,
                positive_class,
                negative_class,
            )
        )

        citations = list(context.sources)
        if item.retention is not None:
            citations.append("supplied selection metadata")
        if item.signed_effect_size is not None:
            citations.append("supplied effect-size metadata")
        citations.extend(term.source for term, _ in matches)
        citations = list(dict.fromkeys(citations))

        narrative = _narrative(
            context,
            item.retention,
            item.signed_effect_size,
            item.effect_metric,
            matches,
            level,
            positive_class,
            negative_class,
        )
        explanations.append(
            BiomarkerExplanation(
                probe_id=context.probe_id,
                gene_symbol=symbol,
                status=context.status,
                direction=direction,
                signed_effect_size=item.signed_effect_size,
                retention_method=(
                    _method_label(item.retention.method) if item.retention else None
                ),
                retention_rank=item.retention.rank if item.retention else None,
                evidence_level=level,
                narrative=narrative,
                reasons=reasons,
                citations=citations,
            )
        )

    themes = [
        f"{name} ({count} biomarkers)"
        for name, count in theme_counts.most_common(5)
        if count >= 2
    ]
    summary_parts = [
        (
            f"{len(explanations)} biomarkers were supplied: {resolved_count} resolved to a gene "
            f"and {pathway_count} had KEGG pathway annotations."
        )
    ]
    if direction_counts:
        parts = [f"{count} {direction}" for direction, count in direction_counts.items()]
        summary_parts.append("Signed effects placed " + " and ".join(parts) + ".")
    enrichment_summary = None
    if terms:
        enrichment_summary = (
            f"{len(terms)} enrichment terms were supplied; {len(matched_terms)} matched a "
            "biomarker pathway or leading-edge gene."
        )

    return XAIReport(
        title=f"XAI biomarker report: {dataset_id}",
        dataset_id=dataset_id,
        positive_class=positive_class,
        negative_class=negative_class,
        generated_at=generated_at or datetime.now(UTC),
        scope=(
            "Deterministic, evidence-grounded explanation of biomarkers retained for a "
            "research feature-selection analysis. It summarizes supplied annotation, "
            "selection, effect-size, and enrichment fields and does not diagnose disease "
            "or recommend treatment."
        ),
        summary=" ".join(summary_parts),
        pathway_themes=themes,
        enrichment_summary=enrichment_summary,
        enrichment_terms=list(terms),
        biomarkers=explanations,
        limitations=[
            "The report is generated deterministically from the supplied fields; no external literature is added.",
            "Selection ranks and effect sizes describe this dataset and are not held-out validation; they do not establish causal or clinical relevance.",
            "Pathway content is limited to the KEGG annotation returned by Bio-RAG.",
            "Enrichment statements describe pathway or leading-edge membership and are direction-checked, but they do not prove that the gene drives the phenotype.",
            "This report is research software output, not a diagnostic device.",
        ],
    )


def load_contexts(path: str | Path) -> list[ProbeGeneContext]:
    """Load a JSON array of probe contexts or a report containing them."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = (
            payload.get("records")
            or payload.get("gene_contexts")
            or payload.get("biomarkers")
            or []
        )
    return [ProbeGeneContext.model_validate(item) for item in payload]


def load_retention(path: str | Path) -> dict[str, RetentionEvidence]:
    """Load a probe-to-retention mapping or a list with ``probe_id`` fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "retention" in payload:
        payload = payload["retention"]
    if isinstance(payload, list):
        mapping: dict[str, RetentionEvidence] = {}
        for item in payload:
            probe_id = str(item["probe_id"])
            mapping[probe_id] = RetentionEvidence.model_validate(
                {key: value for key, value in item.items() if key != "probe_id"}
            )
        return mapping
    return {
        str(probe_id): RetentionEvidence.model_validate(value)
        for probe_id, value in payload.items()
    }


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_genes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [gene.strip() for gene in value.split(";") if gene.strip()]
    if isinstance(value, list):
        return [str(gene).strip() for gene in value if str(gene).strip()]
    return []


def _load_enrichment_csv(path: Path, fdr: float | None) -> list[EnrichmentTerm]:
    if fdr is not None and not 0.0 <= fdr <= 1.0:
        raise ValueError("fdr must be between 0 and 1.")
    terms: list[EnrichmentTerm] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        is_ora = "Adjusted P-value" in columns
        for row in reader:
            name = row.get("Term") or row.get("Name")
            if not name:
                continue
            nes = _as_float(row.get("NES"))
            term_fdr = _as_float(
                row.get("Adjusted P-value") if is_ora else row.get("FDR q-val")
            )
            if fdr is not None and (term_fdr is None or term_fdr > fdr):
                continue
            direction = None
            if not is_ora and nes is not None and nes != 0:
                direction = (
                    "up_in_positive_class" if nes > 0 else "down_in_positive_class"
                )
            terms.append(
                EnrichmentTerm(
                    term=name,
                    nes=nes,
                    fdr=term_fdr,
                    direction=direction,
                    lead_genes=_split_genes(
                        row.get("Genes") if is_ora else row.get("Lead_genes")
                    ),
                    source=(
                        "over-representation analysis" if is_ora else "GSEA prerank"
                    ),
                )
            )
    terms.sort(
        key=lambda item: (item.fdr is None, item.fdr or 0.0, -abs(item.nes or 0.0))
    )
    return terms


def load_enrichment_terms(
    path: str | Path, *, fdr: float | None = None
) -> list[EnrichmentTerm]:
    """Load enrichment terms from a GSEA report, plain list, or result CSV.

    CSVs accept gseapy prerank or ORA output and can be filtered by ``fdr``.
    JSON reports keep their recorded top terms unless ``fdr`` is supplied.
    """
    source_path = Path(path)
    if source_path.suffix.casefold() == ".csv":
        return _load_enrichment_csv(source_path, fdr)
    if fdr is not None and not 0.0 <= fdr <= 1.0:
        raise ValueError("fdr must be between 0 and 1.")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    records: list[tuple[dict[str, Any], str]] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("prerank"), dict):
            records.extend(
                (record, "GSEA prerank")
                for record in payload["prerank"].get("top_terms", [])
            )
        if isinstance(payload.get("ora"), dict):
            records.extend(
                (record, "over-representation analysis")
                for record in payload["ora"].get("top_terms", [])
            )
        if not records:
            records.extend(
                (record, "supplied enrichment summary")
                for record in payload.get("terms", [])
            )
    else:
        records.extend(
            (record, "supplied enrichment summary") for record in payload
        )
    parsed: list[EnrichmentTerm] = []
    for record, source in records:
        term = record.get("term") or record.get("Term")
        if not term:
            continue
        term_fdr = _as_float(
            record.get("fdr", record.get("FDR q-val", record.get("adjusted_p")))
        )
        if fdr is not None and (term_fdr is None or term_fdr > fdr):
            continue
        parsed.append(
            EnrichmentTerm(
                term=str(term),
                nes=_as_float(record.get("nes", record.get("NES"))),
                fdr=term_fdr,
                direction=record.get("direction"),
                lead_genes=_split_genes(
                    record.get("lead_genes") or record.get("genes")
                ),
                source=record.get("source", source),
            )
        )
    return parsed


def write_report(
    report: XAIReport, *, markdown_path: str | Path, json_path: str | Path
) -> None:
    """Write Markdown and JSON forms of a report."""
    markdown = Path(markdown_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(report.to_markdown(), encoding="utf-8")
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(report.model_dump_json(indent=2), encoding="utf-8")


__all__ = [
    "BiomarkerEvidence",
    "BiomarkerExplanation",
    "EnrichmentTerm",
    "ExplanationReason",
    "RetentionEvidence",
    "XAIReport",
    "generate_xai_report",
    "load_contexts",
    "load_enrichment_terms",
    "load_retention",
    "write_report",
]
