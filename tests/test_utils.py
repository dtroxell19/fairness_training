"""Unit tests for utils.py — dataloader creation, stratification, validate_metric."""

import numpy as np
import pytest
import torch

from fairness_training.utils import (
    create_dataloaders,
    create_stratified_dataloaders,
    get_fairness_metric,
    validate_metric,
)
from fairness_training.fairness_metrics import (
    MeanPredictionParity,
    MeanResidualFairness,
    EqualizedOdds,
    FairnessMetric,
)
import cvxpy as cp


# ---------------------------------------------------------------------------
# get_fairness_metric
# ---------------------------------------------------------------------------

class TestGetFairnessMetric:
    def test_string_mean_pred(self):
        m = get_fairness_metric("mean_pred", num_protected_attrs=1)
        assert isinstance(m, MeanPredictionParity)

    def test_string_mean_residual(self):
        m = get_fairness_metric("mean_residual", num_protected_attrs=2)
        assert isinstance(m, MeanResidualFairness)
        assert m.num_protected_attrs == 2

    def test_string_equalized_odds(self):
        m = get_fairness_metric("equalized_odds")
        assert isinstance(m, EqualizedOdds)

    def test_instance_passthrough(self):
        metric = MeanPredictionParity(num_protected_attrs=1)
        assert get_fairness_metric(metric) is metric

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown fairness metric"):
            get_fairness_metric("not_a_real_metric")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            get_fairness_metric(42)


# ---------------------------------------------------------------------------
# create_dataloaders
# ---------------------------------------------------------------------------

class TestCreateDataloaders:
    def _data(self, n=100):
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.rand(n).astype(np.float32)
        return X, y

    def test_train_only(self):
        X, y = self._data()
        (loader,) = create_dataloaders(X, y, batch_size_train=32)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[1] == 5
        assert batch_y.dim() == 2  # unsqueezed to (n, 1)

    def test_train_val_test(self):
        X, y = self._data(200)
        X_v, y_v = self._data(50)
        X_t, y_t = self._data(50)
        loaders = create_dataloaders(X, y, X_v, y_v, X_t, y_t, batch_size_train=64)
        assert len(loaders) == 3

    def test_eval_batch_size_override(self):
        X, y = self._data(100)
        X_v, y_v = self._data(100)
        train_loader, val_loader = create_dataloaders(
            X, y, X_v, y_v, batch_size_train=64, batch_size_eval=20
        )
        batch_x, _ = next(iter(val_loader))
        assert batch_x.shape[0] == 20


# ---------------------------------------------------------------------------
# create_stratified_dataloaders
# ---------------------------------------------------------------------------

class TestStratifiedDataloaders:
    def _binary_data(self, n=400, frac_group1=0.3, seed=1):
        rng = np.random.default_rng(seed)
        A = (rng.random(n) < frac_group1).astype(np.float32)
        X = np.concatenate([A[:, None], rng.standard_normal((n, 9)).astype(np.float32)], axis=1)
        y = rng.random(n).astype(np.float32)
        return X, y

    def test_both_groups_in_every_batch(self):
        X, y = self._binary_data(400)
        (loader,) = create_stratified_dataloaders(
            X, y, protected_attr_idx=0, batch_size_train=50
        )
        for batch_x, _ in loader:
            g0 = (batch_x[:, 0] == 0).sum().item()
            g1 = (batch_x[:, 0] == 1).sum().item()
            assert g0 >= 1, "Group 0 absent from batch"
            assert g1 >= 1, "Group 1 absent from batch"

    def test_batch_size_is_exact(self):
        X, y = self._binary_data(400)
        (loader,) = create_stratified_dataloaders(
            X, y, protected_attr_idx=0, batch_size_train=50
        )
        for batch_x, _ in loader:
            assert batch_x.shape[0] == 50

    def test_group_ratio_approximately_preserved(self):
        frac = 0.3
        X, y = self._binary_data(600, frac_group1=frac)
        (loader,) = create_stratified_dataloaders(
            X, y, protected_attr_idx=0, batch_size_train=100
        )
        all_g1 = []
        for batch_x, _ in loader:
            all_g1.append((batch_x[:, 0] == 1).float().mean().item())
        # Per-batch fraction should be close to the dataset fraction
        avg_frac = np.mean(all_g1)
        assert abs(avg_frac - frac) < 0.05, f"Mean group1 fraction {avg_frac} far from {frac}"

    def test_non_binary_attr_raises(self):
        X = np.random.randn(100, 5).astype(np.float32)
        X[:, 0] = np.random.rand(100)  # continuous, not binary
        y = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="not binary"):
            create_stratified_dataloaders(X, y, protected_attr_idx=0, batch_size_train=20)

    def test_dual_attr_four_groups_present(self):
        n = 400
        rng = np.random.default_rng(7)
        A = (rng.random(n) < 0.5).astype(np.float32)
        B = (rng.random(n) < 0.5).astype(np.float32)
        X = np.stack([A, B] + [rng.standard_normal(n).astype(np.float32) for _ in range(8)], axis=1)
        y = rng.random(n).astype(np.float32)
        (loader,) = create_stratified_dataloaders(
            X, y, protected_attr_idx=[0, 1], batch_size_train=100
        )
        for batch_x, _ in loader:
            for a_val in [0, 1]:
                for b_val in [0, 1]:
                    count = ((batch_x[:, 0] == a_val) & (batch_x[:, 1] == b_val)).sum().item()
                    assert count >= 1, f"Group ({a_val},{b_val}) absent from batch"


# ---------------------------------------------------------------------------
# validate_metric
# ---------------------------------------------------------------------------

class TestValidateMetric:
    def test_valid_metric_passes_silently(self):
        validate_metric(MeanPredictionParity(), input_dim=10, batch_size=20)

    def test_valid_residual_metric_passes(self):
        validate_metric(MeanResidualFairness(), input_dim=10, batch_size=20)

    def test_valid_equalized_odds_passes(self):
        validate_metric(EqualizedOdds(), input_dim=10, batch_size=20)

    def test_invalid_metric_raises(self):
        class BadMetric(FairnessMetric):
            requires_targets = False

            def create_constraints(self, yhat, selection_matrices, slack, y=None):
                # DPP violation: multiplying two parameters (slack * slack)
                return [cp.sum(yhat) <= slack * slack]

            def compute_gap(self, predictions, targets, protected_indicators):
                return 0.0

        with pytest.raises(ValueError, match="[Dd][Pp][Pp]|not DPP|constraint"):
            validate_metric(BadMetric(), input_dim=10, batch_size=20)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            validate_metric("not_a_metric", input_dim=10, batch_size=20)
