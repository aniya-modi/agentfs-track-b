# XAI biomarker report: golub-hu6800

_Generated 2026-11-22T06:43:11.800575+00:00._

## Scope

Deterministic, evidence-grounded explanation of biomarkers retained for a research feature-selection analysis. It summarizes supplied annotation, selection, effect-size, and enrichment fields and does not diagnose disease or recommend treatment.

## Summary

3 biomarkers were supplied: 3 resolved to a gene and 2 had KEGG pathway annotations. Signed effects placed 2 higher in AML and 1 higher in ALL.

65 enrichment terms were supplied; 3 matched a biomarker pathway or leading-edge gene.

### Pathway themes

- No pathway theme recurred across the supplied biomarkers.

### Supplied enrichment evidence

| Term | NES | FDR | Direction |
| --- | ---: | ---: | --- |
| Lysosome | 2.63 | <0.001 | higher in AML |
| Spliceosome | -2.43 | <0.001 | higher in ALL |
| Rheumatoid arthritis | 2.41 | <0.001 | higher in AML |
| Legionellosis | 2.37 | <0.001 | higher in AML |
| Cell cycle | -2.36 | <0.001 | higher in ALL |

## Biomarker explanations

### 1. ZYX (`X95735_at`)

**Retention:** mRMR rank 1 | **Direction:** higher in AML | **Effect:** +2.704 | **Evidence completeness:** high

ZYX (X95735_at) was retained by mRMR at rank 1 (requested k=3). The platform annotation maps it to ZYX (NCBI Gene 7791). The supplied signed Cohen's d is +2.704, indicating higher in AML. It is annotated to 2 KEGG pathways, including Focal adhesion, Cytoskeleton in muscle cells. The NCBI Gene summary begins: "Focal adhesions are actin-rich structures that enable cells to adhere to the extracellular matrix and at which protein complexes involved in signal transduction assemble." Evidence completeness is high based on the supplied annotation, pathway, selection, and effect fields. This is a research explanation, not a clinical interpretation.

**Evidence**
- **Retention** (strong): Retained by mRMR rank 1 (requested k=3). _(Source: supplied selection metadata)_
- **Effect** (moderate): Signed Cohen's d is +2.704 (higher in AML). _(Source: supplied effect-size metadata)_
- **Annotation** (strong): The GPL80 probe maps unambiguously to ZYX (NCBI Gene 7791). _(Source: platform annotation)_
- **Pathway** (strong): Annotated to 2 KEGG pathways: hsa04510 (Focal adhesion), hsa04820 (Cytoskeleton in muscle cells). _(Source: KEGG REST API)_
- **Gene Summary** (strong): NCBI Gene summary: "Focal adhesions are actin-rich structures that enable cells to adhere to the extracellular matrix and at which protein complexes involved in signal transduction assemble. Zyxin is a zinc-binding phosphoprotein that concentrates at focal adhesions and along the actin cytoskeleton. Zyxin has an N-terminal proline-rich domain and three LIM domains in its C-terminal half. The proline-rich domain may i..." _(Source: NCBI Gene)_
- **Limitation** (limited): None of the supplied enrichment terms matched this biomarker's pathways or leading-edge genes. _(Source: supplied enrichment summary)_

**Citations:** NCBI GEO GPL80.annot.gz (Hu6800; annotation build 35, 2016-08-09); NCBI Gene E-utilities; KEGG REST API; supplied selection metadata; supplied effect-size metadata

### 2. CYTH1 (`M85169_at`)

**Retention:** mRMR rank 2 | **Direction:** higher in ALL | **Effect:** -0.896 | **Evidence completeness:** high

CYTH1 (M85169_at) was retained by mRMR at rank 2 (requested k=3). The platform annotation maps it to CYTH1 (NCBI Gene 9267). The supplied signed Cohen's d is -0.896, indicating higher in ALL. It is annotated to 5 KEGG pathways, including Phospholipase D signaling pathway, Endocytosis. The supplied enrichment term 'Pathogenic Escherichia coli infection' links to this biomarker through its pathway annotation 'pathogenic escherichia coli infection' (NES=1.69, FDR=0.021, term direction: higher in AML). Its direction is opposite to this gene's effect, so it indicates pathway membership rather than concordant support. The NCBI Gene summary begins: "The protein encoded by this gene is a member of the PSCD family." Evidence completeness is high based on the supplied annotation, pathway, selection, and effect fields. This is a research explanation, not a clinical interpretation.

**Evidence**
- **Retention** (strong): Retained by mRMR rank 2 (requested k=3). _(Source: supplied selection metadata)_
- **Effect** (moderate): Signed Cohen's d is -0.896 (higher in ALL). _(Source: supplied effect-size metadata)_
- **Annotation** (strong): The GPL80 probe maps unambiguously to CYTH1 (NCBI Gene 9267). _(Source: platform annotation)_
- **Pathway** (strong): Annotated to 5 KEGG pathways: hsa04072 (Phospholipase D signaling pathway), hsa04144 (Endocytosis), hsa05130 (Pathogenic Escherichia coli infection), hsa05131 (Shigellosis), hsa05132 (Salmonella infection). _(Source: KEGG REST API)_
- **Gene Summary** (strong): NCBI Gene summary: "The protein encoded by this gene is a member of the PSCD family. Members of this family have identical structural organization that consists of an N-terminal coiled-coil motif, a central Sec7 domain, and a C-terminal pleckstrin homology (PH) domain. The coiled-coil motif is involved in homodimerization, the Sec7 domain contains guanine-nucleotide exchange protein activity, and the PH domain intera..." _(Source: NCBI Gene)_
- **Enrichment** (moderate): The supplied term 'Pathogenic Escherichia coli infection' links to this biomarker through pathway annotation 'pathogenic escherichia coli infection', NES=1.69, FDR=0.021; term direction: higher in AML. Its direction is opposite to this gene's effect, so it indicates pathway membership rather than concordant support. _(Source: GSEA prerank)_
- **Enrichment** (moderate): The supplied term 'Shigellosis' links to this biomarker through pathway annotation 'shigellosis', NES=1.60, FDR=0.0395; term direction: higher in AML. Its direction is opposite to this gene's effect, so it indicates pathway membership rather than concordant support. _(Source: GSEA prerank)_

**Citations:** NCBI GEO GPL80.annot.gz (Hu6800; annotation build 35, 2016-08-09); NCBI Gene E-utilities; KEGG REST API; supplied selection metadata; supplied effect-size metadata; GSEA prerank

### 3. SRGN (`X17042_at`)

**Retention:** mRMR rank 3 | **Direction:** higher in AML | **Effect:** +2.233 | **Evidence completeness:** moderate

SRGN (X17042_at) was retained by mRMR at rank 3 (requested k=3). The platform annotation maps it to SRGN (NCBI Gene 5552). The supplied signed Cohen's d is +2.233, indicating higher in AML. No KEGG pathway membership was returned for this gene. The NCBI Gene summary begins: "This gene encodes a protein best known as a hematopoietic cell granule proteoglycan." Evidence completeness is moderate based on the supplied annotation, pathway, selection, and effect fields. This is a research explanation, not a clinical interpretation.

**Evidence**
- **Retention** (strong): Retained by mRMR rank 3 (requested k=3). _(Source: supplied selection metadata)_
- **Effect** (moderate): Signed Cohen's d is +2.233 (higher in AML). _(Source: supplied effect-size metadata)_
- **Annotation** (strong): The GPL80 probe maps unambiguously to SRGN (NCBI Gene 5552). _(Source: platform annotation)_
- **Limitation** (limited): No KEGG pathway membership was returned for this gene. _(Source: KEGG REST API)_
- **Gene Summary** (strong): NCBI Gene summary: "This gene encodes a protein best known as a hematopoietic cell granule proteoglycan. Proteoglycans stored in the secretory granules of many hematopoietic cells also contain a protease-resistant peptide core, which may be important for neutralizing hydrolytic enzymes. This encoded protein was found to be associated with the macromolecular complex of granzymes and perforin, which may serve as a medi..." _(Source: NCBI Gene)_

**Citations:** NCBI GEO GPL80.annot.gz (Hu6800; annotation build 35, 2016-08-09); NCBI Gene E-utilities; KEGG REST API; supplied selection metadata; supplied effect-size metadata

## Limitations

- The report is generated deterministically from the supplied fields; no external literature is added.
- Selection ranks and effect sizes describe this dataset and are not held-out validation; they do not establish causal or clinical relevance.
- Pathway content is limited to the KEGG annotation returned by Bio-RAG.
- Enrichment statements describe pathway or leading-edge membership and are direction-checked, but they do not prove that the gene drives the phenotype.
- This report is research software output, not a diagnostic device.
