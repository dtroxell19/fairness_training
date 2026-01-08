"""
Fair Neural Networks

A PyTorch package for training neural networks with differentiable optimization layers 
to enforce fairness constraints.

Usage:
    from fair_nn import FairModel, FairTrainer, create_dataloaders
    
    # Create model with built-in metric
    model = FairModel(
        input_dim=20, 
        hidden_dims=[64, 32], 
        protected_attr_idx=[0, 1],
        fairness_tolerance=0.05,
        fairness_metric='mean_pred'  # or 'mean_residual'
    )
    
    # Or with custom metric
    from fair_nn import FairnessMetric, MeanPredictionParity
    custom_metric = MeanPredictionParity(num_protected_attrs=2)
    model = FairModel(..., fairness_metric=custom_metric)
    
    # Train (large batches with hard constraints)
    trainer = FairTrainer(model, criterion, optimizer)
    history = trainer.fit(train_loader, val_loader, epochs=100)
    
    # Evaluate (can use small batches with online primal-dual updates)
    metrics = trainer.evaluate(test_loader)
"""

from .fair_model import FairModel
from .fair_trainer import (
    FairTrainer, 
    create_dataloaders, 
    create_stratified_dataloaders
)
from .fairness_metrics import (
    FairnessMetric,
    MeanPredictionParity,
    MeanResidualFairness,
    EqualizedOdds,
    get_fairness_metric
)
from .utils import (
    get_fairness_metric,
    create_dataloaders,
    create_stratified_dataloaders
)

__version__ = "0.3.0"

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
    "create_stratified_dataloaders"
    ]