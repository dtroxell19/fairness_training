"""Unit tests for fair_model.py — initialization, validation, forward pass, checkpoint."""

import os
import tempfile
import warnings
import numpy as np
import pytest
import torch
import torch.nn as nn

from fairness_training import FairModel
from fairness_training.fairness_metrics import MeanPredictionParity


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestFairModelInit:
    def test_invalid_prediction_bounds_raises(self):
        with pytest.raises(ValueError, match="lb < ub"):
            FairModel(input_dim=10, hidden_dims=[8], prediction_bounds=(1.0, 0.0))

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match="lb < ub"):
            FairModel(input_dim=10, hidden_dims=[8], prediction_bounds=(0.5, 0.5))

    def test_more_than_two_protected_attrs_raises(self):
        with pytest.raises(ValueError, match="up to 2 protected attributes"):
            FairModel(input_dim=10, hidden_dims=[8], protected_attr_idx=[0, 1, 2])

    def test_single_int_protected_attr_normalized_to_list(self):
        model = FairModel(input_dim=10, hidden_dims=[8], protected_attr_idx=3)
        assert model.protected_attr_idx == [3]

    def test_custom_network_no_hidden_dims_required(self):
        net = nn.Linear(10, 1)
        model = FairModel(input_dim=10, custom_network=net, protected_attr_idx=0)
        assert model.ffnn is net

    def test_no_custom_network_and_no_hidden_dims_raises(self):
        with pytest.raises(ValueError):
            FairModel(input_dim=10, protected_attr_idx=0)

    def test_string_metric_resolved(self):
        model = FairModel(input_dim=10, hidden_dims=[8], fairness_metric="mean_pred")
        assert isinstance(model.fairness_metric, MeanPredictionParity)

    def test_metric_instance_accepted(self):
        metric = MeanPredictionParity(num_protected_attrs=1)
        model = FairModel(input_dim=10, hidden_dims=[8], fairness_metric=metric)
        assert model.fairness_metric is metric

    def test_deprecated_fairness_criterion_warns(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            FairModel(
                input_dim=10, hidden_dims=[8],
                fairness_criterion="mean_pred"
            )


# ---------------------------------------------------------------------------
# forward() — binary attribute validation warning
# ---------------------------------------------------------------------------

class TestFairModelForward:
    def _make_model(self, b_tau=50):
        return FairModel(
            input_dim=4, hidden_dims=[8], protected_attr_idx=0,
            fairness_tolerance=0.05, b_tau=b_tau
        )

    def test_warns_on_non_binary_protected_attr(self):
        model = self._make_model()
        # Column 0 contains 0.5 — not binary
        x = torch.tensor([[0.5, 1, 0, 0], [1.0, 0, 1, 0]], dtype=torch.float32)
        with pytest.warns(UserWarning, match="other than 0 and 1"):
            try:
                model(x, inference=True)
            except Exception:
                pass  # solver may fail on continuous attr; warning is what we care about

    def test_no_warning_on_binary_attr(self):
        model = self._make_model()
        # balanced batch needed for solver
        x = torch.zeros(10, 4)
        x[:5, 0] = 0.0
        x[5:, 0] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                model(x, inference=True)
            except UserWarning:
                pytest.fail("Unexpected UserWarning for valid binary attr")
            except Exception:
                pass  # solver failure OK in unit test context

    def test_requires_targets_enforced(self):
        model = FairModel(
            input_dim=4, hidden_dims=[8], protected_attr_idx=0,
            fairness_metric="mean_residual", b_tau=50
        )
        x = torch.zeros(10, 4)
        x[5:, 0] = 1.0
        with pytest.raises(ValueError, match="Target y must be provided"):
            model(x, inference=True)


# ---------------------------------------------------------------------------
# FairModel.wrap()
# ---------------------------------------------------------------------------

class TestFairModelWrap:
    def test_wrap_infers_dims(self):
        net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        model = FairModel.wrap(net, protected_attr_idx=0, input_dim=10)
        assert model.input_dim == 10
        assert model.output_dim == 1
        assert model.ffnn is net

    def test_wrap_custom_tolerance(self):
        net = nn.Linear(5, 1)
        model = FairModel.wrap(net, protected_attr_idx=[0, 1], fairness_tolerance=0.02, input_dim=5)
        assert model.fairness_tolerance == 0.02
        assert model.protected_attr_idx == [0, 1]

    def test_wrap_without_input_dim_raises(self):
        net = nn.Linear(5, 1)
        with pytest.raises((ValueError, TypeError)):
            FairModel.wrap(net, protected_attr_idx=0)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def _make_trainer(self):
        from fairness_training import FairTrainer
        import torch.optim as optim
        model = FairModel(
            input_dim=10, hidden_dims=[8], protected_attr_idx=0,
            fairness_tolerance=0.05, b_tau=50
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        return FairTrainer(model, criterion, optimizer)

    def test_checkpoint_saves_model_config(self):
        trainer = self._make_trainer()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            checkpoint = torch.load(path, map_location="cpu")
            assert "model_config" in checkpoint
            cfg = checkpoint["model_config"]
            assert cfg["protected_attr_idx"] == [0]
            assert cfg["fairness_tolerance"] == pytest.approx(0.05)
            assert cfg["b_tau"] == 50
            assert cfg["fairness_metric"] == "MeanPredictionParity"
        finally:
            os.unlink(path)

    def test_checkpoint_load_warns_on_config_mismatch(self):
        trainer = self._make_trainer()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)

            # Create new trainer with different b_tau
            import torch.optim as optim
            model2 = FairModel(
                input_dim=10, hidden_dims=[8], protected_attr_idx=0,
                fairness_tolerance=0.05, b_tau=999  # different
            )
            from fairness_training import FairTrainer as _FairTrainer
            trainer2 = _FairTrainer(
                model2, nn.MSELoss(), optim.Adam(model2.parameters())
            )
            with pytest.warns(UserWarning, match="Checkpoint config does not match"):
                trainer2.load_checkpoint(path)
        finally:
            os.unlink(path)

    def test_checkpoint_roundtrip_weights(self):
        trainer = self._make_trainer()
        # Record weight before saving
        param_before = next(trainer.model.parameters()).detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)

            # Perturb weights
            with torch.no_grad():
                next(trainer.model.parameters()).fill_(99.0)

            trainer.load_checkpoint(path)
            param_after = next(trainer.model.parameters()).detach()
            assert torch.allclose(param_before, param_after)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# reset_inference_state
# ---------------------------------------------------------------------------

def test_reset_inference_state_zeros_all_fields():
    model = FairModel(input_dim=10, hidden_dims=[8], protected_attr_idx=0, b_tau=50)
    model.lambda_dual = 5.0
    model.dual_update_count = 10
    model.cumulative_samples = 500
    model.cumulative_weighted_violation = 2.5
    model.lambda_max = 7.0
    model.reset_inference_state()
    assert model.lambda_dual == 0.0
    assert model.dual_update_count == 0
    assert model.cumulative_samples == 0
    assert model.cumulative_weighted_violation == 0.0
    assert model.lambda_max == 0.0
