"""
Training utilities for fair neural networks.

Provides high-level training interface with built-in early stopping, 
learning rate scheduling, and fairness monitoring.

Key Features:
- Training always uses hard per-batch fairness constraints
- Automatic inference state reset before validation/test
- Aggregate fairness tracking during evaluation
- Support for stratified batching to maintain group ratios
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, Tuple
import numpy as np


class FairTrainer:
    """
    Trainer for FairModel with fairness monitoring.
    
    Handles the training loop including:
    - Training with hard per-batch fairness constraints
    - Automatic inference state reset before validation
    - Aggregate fairness tracking and reporting
    - Early stopping based on validation loss
    
    Args:
        model: FairModel instance to train
        criterion: Loss function (e.g., nn.MSELoss())
        optimizer: Optimizer (e.g., Adam)
        device: Device to train on ('cpu' or 'cuda')
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Epochs to wait before early stopping
        early_stopping_delta: Minimum improvement to reset patience
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[object] = None,
        early_stopping_patience: int = 25,
        early_stopping_delta: float = 1e-5
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        # Early stopping
        self.patience = early_stopping_patience
        self.delta = early_stopping_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_fairness_gap': [],
            'val_fairness_gap': [],
            'learning_rate': []
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: int = 1,
        log_interval: int = 10
    ) -> Dict:
        """
        Driver function that trains model.
        
        Training uses hard per-batch fairness constraints.
        Validation uses the same batch size regime as the loader provides.
        
        Args:
            train_loader: Training data loader (batch_size should be >= b_tau)
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            log_interval: Print every N epochs
            
        Returns:
            Dictionary containing training history
        """
        for epoch in range(epochs):
            # Training phase (always uses hard constraints)
            train_loss, train_gap = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_fairness_gap'].append(train_gap)
            
            # Validation phase
            if val_loader is not None:
                # Reset inference state before validation
                self.model.reset_inference_state()
                
                val_loss, val_gap = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_fairness_gap'].append(val_gap)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Early stopping check
                if val_loss < self.best_val_loss - self.delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    self.patience_counter += 1
                
                # Logging
                if verbose >= 1 and (epoch + 1) % log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f"Epoch {epoch+1:4d} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Train Gap: {train_gap:.6f} | "
                        f"Val Gap: {val_gap:.6f}"
                    )
                    if verbose >= 2:
                        print(f"           | LR: {lr:.6f}")
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    if verbose >= 1:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    break
            else:
                # No validation set - just log training
                if verbose >= 1 and (epoch + 1) % log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f"Epoch {epoch+1:4d} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Train Gap: {train_gap:.6f} | "
                        f"LR: {lr:.6f}"
                    )
            
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch with hard fairness constraints."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Collect all predictions and attributes for aggregate gap computation
        all_preds = []
        all_targets = []
        all_protected = {attr_idx: [] for attr_idx in self.model.protected_attr_idx}
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Skip if missing a group in any protected attribute
            skip_batch = False
            for attr_idx in self.model.protected_attr_idx:
                if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                   (batch_x[:, attr_idx] == 1).sum() < 1:
                    skip_batch = True
                    break
            
            if skip_batch:
                continue
            
            self.optimizer.zero_grad()
            
            if self.model.requires_targets:
                predictions = self.model(batch_x, y=batch_y, inference=False)
            else:
                predictions = self.model(batch_x, inference=False)
            
            if batch_y.dim() == 1:
                batch_y = batch_y.unsqueeze(1)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()
            

            total_loss += loss.item()
            num_batches += 1
            
            # Collect for aggregate gap computation
            all_preds.append(predictions.detach().cpu())
            all_targets.append(batch_y.detach().cpu())
            for attr_idx in self.model.protected_attr_idx:
                all_protected[attr_idx].append(batch_x[:, attr_idx].detach().cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        aggregate_gap = self._compute_aggregate_gap(all_preds, all_targets, all_protected)
        
        return avg_loss, aggregate_gap
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Collect all predictions and attributes for aggregate gap computation
        all_preds = []
        all_targets = []
        all_protected = {attr_idx: [] for attr_idx in self.model.protected_attr_idx}
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Skip if missing a group in any protected attribute
                skip_batch = False
                for attr_idx in self.model.protected_attr_idx:
                    if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                       (batch_x[:, attr_idx] == 1).sum() < 1:
                        skip_batch = True
                        break
                
                if skip_batch:
                    continue
                
                # Forward pass (inference mode)
                if self.model.requires_targets:
                    predictions = self.model(batch_x, y=batch_y, inference=True)
                else:
                    predictions = self.model(batch_x, inference=True)
                
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect for aggregate gap computation
                all_preds.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
                for attr_idx in self.model.protected_attr_idx:
                    all_protected[attr_idx].append(batch_x[:, attr_idx].cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        aggregate_gap = self._compute_aggregate_gap(all_preds, all_targets, all_protected)
        
        return avg_loss, aggregate_gap
    
    def _compute_aggregate_gap(
        self,
        all_preds: list,
        all_targets: list,
        all_protected: Dict[int, list]
    ) -> float:
        """
        Compute fairness gap over all examples
        
        Args:
            all_preds: List of prediction tensors from each batch
            all_targets: List of target tensors from each batch
            all_protected: Dict mapping attr_idx -> list of protected attribute tensors
            
        Returns:
            Max fairness gap across all protected attributes
        """
        if len(all_preds) == 0:
            return 0.0
        
        # Concatenate all predictions, targets, protected attributes
        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy() if all_targets else None
        
        protected_indicators = {
            attr_idx: torch.cat(tensors, dim=0).numpy()
            for attr_idx, tensors in all_protected.items()
        }
        
        return self.model.fairness_metric.compute_gap(
            preds, targets, protected_indicators
        )

    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Resets inference state before evaluation, then runs inference
        on all batches (using primal-dual for small batches if applicable).
        
        Args:
            test_loader: Test data loader
            return_predictions: If True, return predictions array
            
        Returns:
            Dictionary with test metrics including per-attribute fairness gaps
        """
        self.model.reset_inference_state()
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_protected = {attr_idx: [] for attr_idx in self.model.protected_attr_idx}
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Skip if missing a group
                skip_batch = False
                for attr_idx in self.model.protected_attr_idx:
                    if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                       (batch_x[:, attr_idx] == 1).sum() < 1:
                        skip_batch = True
                        break
                
                if skip_batch:
                    continue
                
                # Forward pass (inference mode)
                if self.model.requires_targets:
                    predictions = self.model(batch_x, y=batch_y, inference=True)
                else:
                    predictions = self.model(batch_x, inference=True)
                
                # Compute loss
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions + protected attr
                all_preds.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
                for attr_idx in self.model.protected_attr_idx:
                    all_protected[attr_idx].append(batch_x[:, attr_idx].cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        for attr_idx in self.model.protected_attr_idx:
            all_protected[attr_idx] = torch.cat(all_protected[attr_idx], dim=0)
        
        # Compute metrics
        metrics = {
            'test_loss': total_loss / max(num_batches, 1)
        }
        
        protected_indicators = {
            attr_idx: all_protected[attr_idx].numpy()
            for attr_idx in self.model.protected_attr_idx
        }
        
        aggregate_gap = self.model.fairness_metric.compute_gap(
            all_preds.numpy(),
            all_targets.numpy(),
            protected_indicators
        )
        
        # Also compute per-attribute gaps for detailed reporting
        max_gap = 0.0
        for attr_idx in self.model.protected_attr_idx:
            protected = all_protected[attr_idx]
            mask_0 = protected == 0
            mask_1 = protected == 1
            
            if mask_0.sum() == 0 or mask_1.sum() == 0:
                continue
            
            # Compute per-attribute gap using the metric
            single_attr_indicators = {attr_idx: protected.numpy()}
            attr_gap = self.model.fairness_metric.compute_gap(
                all_preds.numpy(),
                all_targets.numpy(),
                single_attr_indicators
            )
            metrics[f'fairness_gap_attr_{attr_idx}'] = attr_gap
            max_gap = max(max_gap, attr_gap)
        
        # Overall fairness gap (max across all attributes)
        metrics['fairness_gap'] = max_gap
        
        # Add inference statistics (for small-batch regime)
        metrics['lambda_max'] = self.model.lambda_max
        metrics['total_inference_samples'] = self.model.cumulative_samples
    
        if return_predictions:
            metrics['predictions'] = all_preds.numpy()
            metrics['targets'] = all_targets.numpy()
            metrics['protected'] = {
                attr_idx: all_protected[attr_idx].numpy() 
                for attr_idx in self.model.protected_attr_idx
            }
        
        return metrics
    
    def save_checkpoint(self, filepath: str, include_history: bool = True):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        
        if include_history:
            checkpoint['history'] = self.history
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"Checkpoint loaded from {filepath}")
