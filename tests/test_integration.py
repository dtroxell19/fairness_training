"""Integration tests — full training loop on tiny synthetic data.

Marked @pytest.mark.slow; run with:  pytest -m slow  or  pytest  (runs all)
Fast subset (no solver): pytest -m "not slow"
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from fairness_training import FairModel, FairTrainer
from fairness_training.utils import create_stratified_dataloaders
from fairness_training.fairness_metrics import MeanPredictionParity


def _make_synthetic(n=300, n_features=10, seed=42):
    rng = np.random.default_rng(seed)
    A = (rng.random(n) < 0.4).astype(np.float32)
    X = np.concatenate([A[:, None], rng.standard_normal((n, n_features - 1)).astype(np.float32)], axis=1)
    y = (rng.random(n) > 0.5).astype(np.float32)
    return X, y


@pytest.mark.slow
class TestEndToEndTraining:
    """Full training + evaluation with tiny data and small b_tau."""

    def setup_method(self):
        X, y = _make_synthetic(n=300)
        split = 240
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_val, self.y_val = X[split:], y[split:]

    def _make_loaders(self, batch_size=60):
        return create_stratified_dataloaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            protected_attr_idx=0,
            batch_size_train=batch_size,
            batch_size_eval=batch_size,
        )

    def _make_model(self, **kwargs):
        return FairModel(
            input_dim=10, hidden_dims=[16, 8], protected_attr_idx=0,
            fairness_tolerance=0.05, b_tau=60, **kwargs
        )

    def test_training_converges_without_error(self):
        train_loader, val_loader = self._make_loaders()
        model = self._make_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = FairTrainer(model, nn.MSELoss(), optimizer, early_stopping_patience=5)
        history = trainer.fit(train_loader, val_loader, epochs=5, verbose=0)
        assert len(history["train_loss"]) >= 1
        assert all(np.isfinite(v) for v in history["train_loss"])

    def test_fairness_gap_at_or_near_tolerance(self):
        train_loader, val_loader = self._make_loaders()
        model = self._make_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = FairTrainer(model, nn.MSELoss(), optimizer)
        trainer.fit(train_loader, val_loader, epochs=10, verbose=0)
        metrics = trainer.evaluate(val_loader)
        # Gap should be within 10% of tolerance (allowing for small-sample variance)
        assert metrics["fairness_gap"] <= model.fairness_tolerance * 1.10

    def test_weighted_avg_gap_reported(self):
        train_loader, val_loader = self._make_loaders()
        model = self._make_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = FairTrainer(model, nn.MSELoss(), optimizer)
        trainer.fit(train_loader, val_loader, epochs=5, verbose=0)
        metrics = trainer.evaluate(val_loader)
        assert "weighted_avg_fairness_gap" in metrics
        assert np.isfinite(metrics["weighted_avg_fairness_gap"])

    def test_custom_network_integration(self):
        """FairModel.wrap() with a custom nn.Module trains without error."""
        net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        model = FairModel.wrap(
            net,
            protected_attr_idx=0,
            fairness_tolerance=0.05,
            b_tau=60,
            input_dim=10,
            prediction_bounds=(0.0, 1.0),
        )
        train_loader, val_loader = self._make_loaders()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = FairTrainer(model, nn.MSELoss(), optimizer)
        history = trainer.fit(train_loader, val_loader, epochs=3, verbose=0)
        assert len(history["train_loss"]) >= 1

    def test_mean_residual_metric_trains(self):
        X, y = _make_synthetic(n=300)
        # Continuous targets for regression
        y = y + np.random.default_rng(0).standard_normal(300).astype(np.float32) * 0.1
        split = 240
        train_loader, val_loader = create_stratified_dataloaders(
            X[:split], y[:split], X[split:], y[split:],
            protected_attr_idx=0, batch_size_train=60, batch_size_eval=60,
        )
        model = FairModel(
            input_dim=10, hidden_dims=[16, 8], protected_attr_idx=0,
            fairness_tolerance=0.10, b_tau=60, fairness_metric="mean_residual",
            prediction_bounds=(-3.0, 3.0),
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = FairTrainer(model, nn.MSELoss(), optimizer)
        history = trainer.fit(train_loader, val_loader, epochs=3, verbose=0)
        assert len(history["train_loss"]) >= 1

    def test_evaluate_returns_expected_keys(self):
        train_loader, val_loader = self._make_loaders()
        model = self._make_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = FairTrainer(model, nn.MSELoss(), optimizer)
        trainer.fit(train_loader, val_loader, epochs=3, verbose=0)
        metrics = trainer.evaluate(val_loader)
        for key in ["test_loss", "fairness_gap", "pooled_fairness_gap",
                    "weighted_avg_fairness_gap", "lambda_max"]:
            assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Weighted-average gap math verification
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_weighted_avg_gap_formula():
    """
    Manually simulate two primal-dual batches and verify the formula:
        weighted_avg_gap = cumulative_weighted_violation / cumulative_samples + epsilon
    matches the true batch-size-weighted average of per-batch gaps.
    """
    model = FairModel(
        input_dim=4, hidden_dims=[8], protected_attr_idx=0,
        fairness_tolerance=0.05, b_tau=999,  # force primal-dual for all batches
    )
    model.reset_inference_state()
    model.eval()

    batches = []
    manual_gaps = []
    manual_sizes = []

    for seed in [0, 1]:
        rng = np.random.default_rng(seed)
        n = 40
        A = np.array([0]*20 + [1]*20, dtype=np.float32)
        x = torch.from_numpy(np.concatenate([A[:, None], rng.standard_normal((n, 3)).astype(np.float32)], axis=1))
        with torch.no_grad():
            pred = model(x, inference=True)
        preds_np = pred.squeeze().detach().numpy()
        gap = abs(preds_np[:20].mean() - preds_np[20:].mean())
        manual_gaps.append(gap)
        manual_sizes.append(n)

    N = sum(manual_sizes)
    expected_weighted_avg = sum(g * s for g, s in zip(manual_gaps, manual_sizes)) / N
    computed = model.cumulative_weighted_violation / model.cumulative_samples + model.fairness_tolerance

    assert computed == pytest.approx(expected_weighted_avg, abs=1e-5)
