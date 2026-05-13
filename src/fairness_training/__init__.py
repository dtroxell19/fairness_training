"""
Fairness Layers

A PyTorch package for training neural networks with differentiable optimization layers 
to enforce fairness constraints.

Usage:
    from fairness_training import FairModel, FairTrainer, create_dataloaders
    
    # Create model with built-in metric
    model = FairModel(
        input_dim=20, 
        hidden_dims=[64, 32], 
        protected_attr_idx=[0, 1],
        fairness_tolerance=0.05,
        fairness_metric='mean_pred'  # or 'mean_residual'
    )
    
    # Or with custom metric
    from fairness_training import FairnessMetric, MeanPredictionParity
    custom_metric = MeanPredictionParity(num_protected_attrs=2)
    model = FairModel(..., fairness_metric=custom_metric)
    
    # Train (large batches with hard constraints)
    trainer = FairTrainer(model, criterion, optimizer)
    history = trainer.fit(train_loader, val_loader, epochs=100)
    
    # Evaluate (can use small batches with online primal-dual updates)
    metrics = trainer.evaluate(test_loader)
"""

from .fair_model import FairModel
from .fair_trainer import FairTrainer
from .fairness_metrics import (
    FairnessMetric,
    MeanPredictionParity,
    MeanResidualFairness,
    EqualizedOdds,
)
from .utils import (
    get_fairness_metric,
    create_dataloaders,
    create_stratified_dataloaders,
    validate_metric,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "FairModel",
    "FairTrainer",
    # Fairness metrics (extensible)
    "FairnessMetric",
    "MeanPredictionParity",
    "MeanResidualFairness",
    "EqualizedOdds",
    "get_fairness_metric",
    # Data loading
    "create_dataloaders",
    "create_stratified_dataloaders",
    # Utilities
    "validate_metric",
]