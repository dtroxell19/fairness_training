"""Unit tests for fairness_metrics.py — pure numpy, no torch/cvxpy required."""

import numpy as np
import pytest
import torch

from fairness_training.fairness_metrics import (
    MeanPredictionParity,
    MeanResidualFairness,
    EqualizedOdds,
    FairnessMetric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indicators(n, frac_group1=0.4, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < frac_group1).astype(np.float32)


# ---------------------------------------------------------------------------
# MeanPredictionParity.compute_gap
# ---------------------------------------------------------------------------

class TestMeanPredictionParityGap:
    metric = MeanPredictionParity(num_protected_attrs=1)

    def test_zero_gap_when_means_equal(self):
        preds = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        indicators = {0: np.array([0, 0, 1, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, None, indicators) == pytest.approx(0.0)

    def test_exact_gap(self):
        # group 0 mean = 0.2, group 1 mean = 0.6  → gap = 0.4
        preds = np.array([0.2, 0.2, 0.6, 0.6], dtype=np.float32)
        indicators = {0: np.array([0, 0, 1, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, None, indicators) == pytest.approx(0.4, abs=1e-6)

    def test_gap_is_absolute(self):
        # group 0 mean > group 1 mean — gap still positive
        preds = np.array([0.8, 0.8, 0.2, 0.2], dtype=np.float32)
        indicators = {0: np.array([0, 0, 1, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, None, indicators) == pytest.approx(0.6, abs=1e-6)

    def test_max_gap_over_multiple_attrs(self):
        # attr 0 (indicator [0,1,0,1]): group0 mean=(0.6+0.3)/2=0.45, group1 mean=(0.5+0.2)/2=0.35 → gap=0.1
        # attr 1 (indicator [0,0,1,1]): group0 mean=(0.6+0.5)/2=0.55, group1 mean=(0.3+0.2)/2=0.25 → gap=0.3
        # max → 0.3
        preds = np.array([0.6, 0.5, 0.3, 0.2], dtype=np.float32)
        indicators = {
            0: np.array([0, 1, 0, 1], dtype=np.float32),
            1: np.array([0, 0, 1, 1], dtype=np.float32),
        }
        gap = MeanPredictionParity(num_protected_attrs=2).compute_gap(preds, None, indicators)
        assert gap == pytest.approx(0.3, abs=1e-5)

    def test_skips_missing_group(self):
        # all group 0 — gap should be 0 (no group 1 to compare)
        preds = np.array([0.3, 0.7], dtype=np.float32)
        indicators = {0: np.array([0, 0], dtype=np.float32)}
        assert self.metric.compute_gap(preds, None, indicators) == pytest.approx(0.0)

    def test_accepts_2d_predictions(self):
        preds = np.array([[0.2], [0.2], [0.6], [0.6]], dtype=np.float32)
        indicators = {0: np.array([0, 0, 1, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, None, indicators) == pytest.approx(0.4, abs=1e-6)


# ---------------------------------------------------------------------------
# MeanResidualFairness.compute_gap
# ---------------------------------------------------------------------------

class TestMeanResidualFairnessGap:
    metric = MeanResidualFairness(num_protected_attrs=1)

    def test_zero_gap_when_residuals_equal(self):
        # residual = y - y_hat; both groups have mean residual 0
        y = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        preds = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        indicators = {0: np.array([0, 0, 1, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, y, indicators) == pytest.approx(0.0)

    def test_exact_residual_gap(self):
        # group 0: residual = 1.0 - 0.5 = 0.5; group 1: residual = 0.0
        y = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        preds = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        indicators = {0: np.array([0, 0, 1, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, y, indicators) == pytest.approx(0.5, abs=1e-6)

    def test_raises_without_targets(self):
        with pytest.raises(ValueError, match="requires targets"):
            self.metric.compute_gap(np.zeros(4), None, {0: np.array([0, 0, 1, 1])})

    def test_gap_is_max_over_groups(self):
        # group 0 residual = 0.3, group 1 residual = 0.1 → max = 0.3
        y = np.array([0.8, 0.8, 0.6, 0.6], dtype=np.float32)
        preds = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        indicators = {0: np.array([0, 0, 1, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, y, indicators) == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# EqualizedOdds.compute_gap
# ---------------------------------------------------------------------------

class TestEqualizedOddsGap:
    metric = EqualizedOdds(num_protected_attrs=1)

    def test_zero_gap_when_equal_within_outcomes(self):
        preds = np.array([0.4, 0.4, 0.8, 0.8], dtype=np.float32)
        targets = np.array([0, 0, 1, 1], dtype=np.float32)
        indicators = {0: np.array([0, 1, 0, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, targets, indicators) == pytest.approx(0.0)

    def test_gap_on_y1_subgroup(self):
        # Y=1 group: group0 mean = 0.9, group1 mean = 0.5 → gap = 0.4
        preds = np.array([0.3, 0.3, 0.9, 0.5], dtype=np.float32)
        targets = np.array([0, 0, 1, 1], dtype=np.float32)
        indicators = {0: np.array([0, 1, 0, 1], dtype=np.float32)}
        assert self.metric.compute_gap(preds, targets, indicators) == pytest.approx(0.4, abs=1e-5)

    def test_raises_without_targets(self):
        with pytest.raises(ValueError, match="requires targets"):
            self.metric.compute_gap(np.zeros(4), None, {0: np.array([0, 0, 1, 1])})


# ---------------------------------------------------------------------------
# Selection matrix construction
# ---------------------------------------------------------------------------

class TestSelectionMatrices:
    def test_default_two_selectors_per_attr(self):
        x = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float32)
        metric = MeanPredictionParity(num_protected_attrs=2)
        mats = metric.create_selection_matrices(x, None, [0, 1])
        assert len(mats) == 4  # 2 groups × 2 attrs
        np.testing.assert_array_equal(mats[0], [1, 0, 1])  # attr 0, group 0
        np.testing.assert_array_equal(mats[1], [0, 1, 0])  # attr 0, group 1

    def test_equalized_odds_four_selectors_per_attr(self):
        x = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)
        y = torch.tensor([[0], [0], [1], [1]], dtype=torch.float32)
        metric = EqualizedOdds(num_protected_attrs=1)
        mats = metric.create_selection_matrices(x, y, [0])
        assert len(mats) == 4  # Y=0&A=0, Y=0&A=1, Y=1&A=0, Y=1&A=1
        np.testing.assert_array_equal(mats[0], [1, 0, 0, 0])  # Y=0, A=0
        np.testing.assert_array_equal(mats[3], [0, 0, 0, 1])  # Y=1, A=1


# ---------------------------------------------------------------------------
# requires_targets flags
# ---------------------------------------------------------------------------

def test_requires_targets_flags():
    assert MeanPredictionParity().requires_targets is False
    assert MeanPredictionParity().requires_y_in_constraints is False
    assert MeanResidualFairness().requires_targets is True
    assert MeanResidualFairness().requires_y_in_constraints is True
    assert EqualizedOdds().requires_targets is True
    assert EqualizedOdds().requires_y_in_constraints is False
