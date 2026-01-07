"""
Fair Neural Networks

A PyTorch package for training neural networks with differentiable fairness 
constraints using cvxpylayers.

Key Features:
- Differentiable fairness layer for end-to-end training (Goal G2)
- Training: Hard per-batch constraints (requires large batch size >= b_tau)
- Inference: Hard constraints for large batches, primal-dual algorithm for small batches
- Provable aggregate fairness guarantees via Theorem 2.2 (Goal G1)
- Support for any neural network architecture (Goal G3)
- Multiple protected attributes (marginal fairness)
- Mean prediction and mean residual fairness criteria

Usage:
    from fair_nn import FairModel, FairTrainer, create_dataloaders
    
    # Create model
    model = FairModel(
        input_dim=20, 
        hidden_dims=[64, 32], 
        protected_attr_idx=[0, 1],
        fairness_tolerance=0.05,
        b_tau=2000
    )
    
    # Train (large batches with hard constraints)
    trainer = FairTrainer(model, criterion, optimizer)
    history = trainer.fit(train_loader, val_loader, epochs=100)
    
    # Evaluate (can use small batches with primal-dual)
    metrics = trainer.evaluate(test_loader)
"""

from .fair_model import FairModel
from .trainer import (
    FairTrainer, 
    create_dataloaders, 
    create_stratified_dataloaders
)
from .utils import (
    compute_demographic_parity,
    compute_mean_prediction_parity,
    compute_equalized_residuals,
    compute_equalized_odds,
    compute_all_fairness_metrics,
    prepare_data_with_protected_attr,
    generate_synthetic_fairness_data,
    generate_synthetic_classification_data,
    print_fairness_report,
    check_fairness_constraint
)

__version__ = "0.2.0"

__all__ = [
    # Core classes
    "FairModel",
    "FairTrainer",
    # Data loading
    "create_dataloaders",
    "create_stratified_dataloaders",
    # Fairness metrics
    "compute_demographic_parity",
    "compute_mean_prediction_parity",
    "compute_equalized_residuals",
    "compute_equalized_odds",
    "compute_all_fairness_metrics",
    # Data utilities
    "prepare_data_with_protected_attr",
    "generate_synthetic_fairness_data",
    "generate_synthetic_classification_data",
    # Evaluation utilities
    "print_fairness_report",
    "check_fairness_constraint",
]