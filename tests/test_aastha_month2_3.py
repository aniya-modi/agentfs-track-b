from pathlib import Path
"""End-to-end verification tests for Month 2 & 3 selectors, registry, and CV stability."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agentfs.baseline_selectors import LassoSelector, MRMRSelector
from agentfs.evaluation import (
    jaccard_stability_index,
    kuncheva_stability_index,
    run_repeated_nested_cv,
)
from agentfs.hypergraph_centrality import HypergraphCentralitySelector
from agentfs.schemas import FeatureSelectionConfig
from agentfs.tool_registry import FeatureSelectionToolRegistry


def _make_synthetic_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 80))
    y = np.array([0] * 20 + [1] * 20)
    # Inject strong signal into features 0, 1, 2
    X[20:, :3] += 3.0
    H = sparse.random(80, 10, density=0.2, format="csr", random_state=seed)
    H.data[:] = 1.0
    probe_ids = [f"PROBE_{i}_at" for i in range(80)]
    return X, y, H, probe_ids


def test_selectors_recover_signal_and_match_shapes():
    X, y, H, _ = _make_synthetic_data()
    selectors = [
        MRMRSelector(top_k=5),
        LassoSelector(top_k=5),
        HypergraphCentralitySelector(H=H, top_k=5),
    ]
    for sel in selectors:
        X_out = sel.fit_transform(X, y)
        assert X_out.shape == (40, 5)
        idx = sel.get_support(indices=True)
        mask = sel.get_support(indices=False)
        assert len(idx) == 5
        assert mask.dtype == bool and mask.sum() == 5
        # Must recover at least two of the three true signal probes (0, 1, 2)
        assert len(set(idx) & {0, 1, 2}) >= 2


def test_tool_registry_contract_and_probe_mapping():
    X, y, H, probe_ids = _make_synthetic_data()
    registry = FeatureSelectionToolRegistry(H=H)

    tools = registry.list_tools()
    assert {t["name"] for t in tools} == {"hypergraph_centrality", "mrmr", "lasso"}

    for method in ["hypergraph_centrality", "mrmr", "lasso"]:
        cfg = FeatureSelectionConfig(method=method, k=8, two_stage_cascade=True, knockoff_fdr=0.05)
        res = registry.execute(cfg, X, y, feature_ids=probe_ids)
        assert res.k_selected == 8
        assert len(res.selected_feature_ids) == 8
        assert all(pid.endswith("_at") for pid in res.selected_feature_ids)
        assert res.X_train_selected.shape == (40, 8)

    # Registry without H must raise clear ValueError for hypergraph_centrality
    registry_no_h = FeatureSelectionToolRegistry(H=None)
    with pytest.raises(ValueError, match="requires pathway incidence matrix H"):
        registry_no_h.execute(FeatureSelectionConfig(method="hypergraph_centrality", k=5), X, y)


def test_stability_metrics_and_nested_cv():
    s1 = np.array([0, 1, 2, 3, 4])
    s2 = np.array([0, 1, 2, 3, 4])
    assert kuncheva_stability_index([s1, s2], n_total_features=100) == pytest.approx(1.0)
    assert jaccard_stability_index([s1, s2]) == pytest.approx(1.0)

    X, y, H, _ = _make_synthetic_data()
    res = run_repeated_nested_cv(
        X,
        y,
        selector_factory=lambda: HypergraphCentralitySelector(H=H, top_k=5),
        param_grid={"selector__top_k": [5], "classifier__C": [1.0]},
        n_splits=3,
        n_repeats=2,
        inner_splits=2,
    )
    assert res["mean_balanced_accuracy"] >= 0.90
    assert len(res["selected_indices"]) == 6


def test_real_golub_smoke():
    golub_path = Path("data/reference/golub_hu6800.csv.gz")
    if not golub_path.exists():
        pytest.skip("Golub reference file not present")
    df = pd.read_csv(golub_path, compression="gzip")
    y = (df["target"].astype(str) == "AML").astype(int).to_numpy()
    X_df = df.drop(columns=["target"])
    probe_ids = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)

    H = sparse.random(X.shape[1], 15, density=0.05, format="csr", random_state=42)
    H.data[:] = 1.0
    registry = FeatureSelectionToolRegistry(H=H)

    cfg = FeatureSelectionConfig(method="hypergraph_centrality", k=15, two_stage_cascade=False)
    res = registry.execute(cfg, X, y, feature_ids=probe_ids)
    assert res.k_selected == 15
    assert res.X_train_selected.shape == (72, 15)
