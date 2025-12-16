"""
Fair Neural Networks

A PyTorch package for training neural networks with differentiable fairness 
constraints using cvxpylayers.
"""

from .fair_model import FairModel
from .trainer import FairTrainer, create_dataloaders

__all__ = [
    "FairModel",
    "FairTrainer", 
    "create_dataloaders",
]