"""
Training utilities for fair neural networks.

Provides high-level training interface with built-in early stopping, 
learning rate scheduling, and fairness monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, Callable, Tuple, List
import numpy as np
from pathlib import Path
import json


class FairTrainer:
    """
    Trainer for FairModel with fairness monitoring.
    
    Args:
        model: FairModel instance to train
        criterion: Loss function (e.g., nn.MSELoss())
        optimizer: Optimizer (e.g., Adam)
        device: Device to train on ('cpu' or 'cuda')
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Epochs to wait before early stopping
        early_stopping_delta: Minimum improvement to reset patience
        
    Example:
        >>> model = FairModel(input_dim=10, hidden_dims=[32, 16])
        >>> criterion = nn.MSELoss()
        >>> optimizer = optim.Adam(model.parameters(), lr=0.01)
        >>> trainer = FairTrainer(model, criterion, optimizer)
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[object] = None,
        early_stopping_patience: int = 50,
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
        log_interval: int = 10,
        fairness_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            log_interval: Print every N epochs
            fairness_callback: Optional function to compute custom fairness metrics
            
        Returns:
            Dictionary containing training history
        """
        for epoch in range(epochs):
            # Training phase
            train_loss, train_gap = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_fairness_gap'].append(train_gap)
            
            # Validation phase
            if val_loader is not None:
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
                    self.best_model_state = self.model.state_dict().copy()
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
                        f"Val Gap: {val_gap:.6f} | "
                        f"LR: {lr:.6f}"
                    )
                
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
            
            # Store learning rate
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Custom callback
            if fairness_callback is not None:
                fairness_callback(epoch, self.model, self.history)
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_gap = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Skip if missing a group in ANY protected attribute
            skip_batch = False
            for attr_idx in self.model.protected_attr_idx:
                if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                (batch_x[:, attr_idx] == 1).sum() < 1:
                    skip_batch = True
                    break
            
            if skip_batch:
                continue
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Pass targets if using mean_residual criterion
            if hasattr(self.model, 'fairness_criterion') and \
            self.model.fairness_criterion == 'mean_residual':
                predictions = self.model(batch_x, y=batch_y, inference=False)
            else:
                predictions = self.model(batch_x, inference=False)
            
            # Compute loss
            if batch_y.dim() == 1:
                batch_y = batch_y.unsqueeze(1)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            gap = self._compute_fairness_gap(batch_x, predictions, batch_y)
            total_gap += gap
            num_batches += 1
            self.optimizer.step()
            
            # Track metrics (NOW includes past predictions in gap calculation)
            total_loss += loss.item()
            
            
            
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_gap = total_gap / max(num_batches, 1)
        
        return avg_loss, avg_gap
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_gap = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Skip if missing a group in ANY protected attribute
                skip_batch = False
                for attr_idx in self.model.protected_attr_idx:
                    if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                    (batch_x[:, attr_idx] == 1).sum() < 1:
                        skip_batch = True
                        break
                
                if skip_batch:
                    continue
                
                # Forward pass (inference mode for validation)
                if hasattr(self.model, 'fairness_criterion') and \
                self.model.fairness_criterion == 'mean_residual':
                    predictions = self.model(batch_x, y=batch_y, inference=True)
                else:
                    predictions = self.model(batch_x, inference=True)
                
                # Compute loss
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                loss = self.criterion(predictions, batch_y)
                
                # Track metrics (NOW includes past predictions in gap calculation)
                total_loss += loss.item()
                gap = self._compute_fairness_gap(batch_x, predictions, batch_y)
                total_gap += gap
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_gap = total_gap / max(num_batches, 1)
        
        return avg_loss, avg_gap
            
    def _compute_fairness_gap(
        self,
        x: torch.Tensor,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute fairness gap using the exact predictions that were in the cvxpylayer solve.
        
        For mean_pred: Returns max |E[Ŷ|A=0] - E[Ŷ|A=1]| across attributes
        For mean_residual: Returns max |E[Y-Ŷ|A=0]| across attributes and groups
        """
        # Use the predictions that were actually in the last solve
        if self.model.last_solve_predictions is None or self.model.last_solve_indicators is None:
            # Fallback to just current batch
            all_preds = predictions.cpu()
            indicators = {attr_idx: x[:, attr_idx].cpu().numpy() 
                        for attr_idx in self.model.protected_attr_idx}
        else:
            all_preds = self.model.last_solve_predictions.cpu()
            indicators = self.model.last_solve_indicators
        
        # Check if model uses mean_residual criterion
        use_residual = hasattr(self.model, 'fairness_criterion') and \
                    self.model.fairness_criterion == 'mean_residual'
        
        max_gap = 0.0
        
        #print(len(all_preds))
        for attr_idx in self.model.protected_attr_idx:
            indicator_array = indicators[attr_idx]
            indicator_0 = indicator_array == 0
            indicator_1 = indicator_array == 1
            
            # Skip if either group is empty
            if indicator_0.sum() == 0 or indicator_1.sum() == 0:
                continue
            
            # Convert to torch tensors for indexing
            indicator_0_tensor = torch.from_numpy(indicator_0)
            indicator_1_tensor = torch.from_numpy(indicator_1)
            
            if use_residual and targets is not None:
                # For mean_residual, we'd need targets for all predictions
                # This is complex since we don't store past targets
                # For now, skip or use approximation
                # TODO: Store past targets if using mean_residual
                mean_0 = all_preds[indicator_0_tensor].mean().item()
                mean_1 = all_preds[indicator_1_tensor].mean().item()
                
                gap = abs(mean_0 - mean_1)
            else:
                # Mean prediction fairness: |E[Ŷ | A=0] - E[Ŷ | A=1]|
                mean_0 = all_preds[indicator_0_tensor].mean().item()
                mean_1 = all_preds[indicator_1_tensor].mean().item()
                gap = abs(mean_0 - mean_1)
            
            max_gap = max(max_gap, gap)
        
        return max_gap

    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            return_predictions: If True, return predictions array
            
        Returns:
            Dictionary with test metrics including per-attribute fairness gaps
        """
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
                
                # Skip if missing a group in ANY protected attribute
                skip_batch = False
                for attr_idx in self.model.protected_attr_idx:
                    if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                       (batch_x[:, attr_idx] == 1).sum() < 1:
                        skip_batch = True
                        break
                
                if skip_batch:
                    continue
                
                # Forward pass (inference mode)
                if hasattr(self.model, 'fairness_criterion') and \
                   self.model.fairness_criterion == 'mean_residual':
                    predictions = self.model(batch_x, y=batch_y, inference=True)
                else:
                    predictions = self.model(batch_x, inference=True)
                
                # Compute loss
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions
                all_preds.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
                
                # Collect protected attributes
                for attr_idx in self.model.protected_attr_idx:
                    all_protected[attr_idx].append(batch_x[:, attr_idx].cpu())
        
        # Aggregate results
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        for attr_idx in self.model.protected_attr_idx:
            all_protected[attr_idx] = torch.cat(all_protected[attr_idx], dim=0)
        
        # Compute overall metrics
        metrics = {
            'test_loss': total_loss / max(num_batches, 1)
        }
        
        # Compute per-attribute fairness metrics
        use_residual = hasattr(self.model, 'fairness_criterion') and \
                       self.model.fairness_criterion == 'mean_residual'
        
        max_gap = 0.0
        for attr_idx in self.model.protected_attr_idx:
            protected = all_protected[attr_idx]
            mask_0 = protected == 0
            mask_1 = protected == 1
            
            if mask_0.sum() == 0 or mask_1.sum() == 0:
                continue
            
            if use_residual:
                # Mean residual fairness
                residuals = all_targets - all_preds
                mean_residual_0 = residuals[mask_0].mean().item()
                mean_residual_1 = residuals[mask_1].mean().item()
                
                metrics[f'mean_residual_attr_{attr_idx}_group_0'] = mean_residual_0
                metrics[f'mean_residual_attr_{attr_idx}_group_1'] = mean_residual_1
                metrics[f'fairness_gap_attr_{attr_idx}'] = max(
                    abs(mean_residual_0), abs(mean_residual_1)
                )
                
                max_gap = max(max_gap, metrics[f'fairness_gap_attr_{attr_idx}'])
            else:
                # Mean prediction fairness
                mean_0 = all_preds[mask_0].mean().item()
                mean_1 = all_preds[mask_1].mean().item()
                
                metrics[f'mean_pred_attr_{attr_idx}_group_0'] = mean_0
                metrics[f'mean_pred_attr_{attr_idx}_group_1'] = mean_1
                metrics[f'std_pred_attr_{attr_idx}_group_0'] = all_preds[mask_0].std().item()
                metrics[f'std_pred_attr_{attr_idx}_group_1'] = all_preds[mask_1].std().item()
                metrics[f'fairness_gap_attr_{attr_idx}'] = abs(mean_0 - mean_1)
                
                max_gap = max(max_gap, metrics[f'fairness_gap_attr_{attr_idx}'])
        
        # Overall fairness gap (max across all attributes)
        metrics['fairness_gap'] = max_gap
        
        if return_predictions:
            metrics['predictions'] = all_preds.numpy()
            metrics['targets'] = all_targets.numpy()
            metrics['protected'] = {
                attr_idx: all_protected[attr_idx].numpy() 
                for attr_idx in self.model.protected_attr_idx
            }
        
        return metrics
    
    def save_checkpoint(self, filepath: str, include_history: bool = True):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            include_history: Whether to save training history
        """
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
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
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


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size_train: int = 32,
    batch_size_eval: Optional[int] = None,
    shuffle_train: bool = True
) -> Tuple[DataLoader, ...]:
    """
    Helper function to create PyTorch DataLoaders.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        X_test, y_test: Test data (optional)
        batch_size_train: Batch size for training
        batch_size_eval: Batch size for validation and test (if None, uses batch_size_train)
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of DataLoaders (train, val, test) - only created if data provided
        
    Example:
        # Same batch size for all
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size_train=64
        )
        
        # Different batch sizes
        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size_train=32,
            batch_size_eval=128  # Larger for inference
        )
    """
    # If eval batch size not specified, use training batch size
    if batch_size_eval is None:
        batch_size_eval = batch_size_train
    
    loaders = []
    
    # Training loader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    if y_train_t.dim() == 1:
        y_train_t = y_train_t.unsqueeze(1)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=shuffle_train,
        drop_last=True
    )
    loaders.append(train_loader)
    
    # Validation loader (uses eval batch size)
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        if y_val_t.dim() == 1:
            y_val_t = y_val_t.unsqueeze(1)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size_eval,  # Use eval batch size
            shuffle=False,
            drop_last=True
        )
        loaders.append(val_loader)
    
    # Test loader (uses eval batch size)
    if X_test is not None and y_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        if y_test_t.dim() == 1:
            y_test_t = y_test_t.unsqueeze(1)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size_eval,  # Use eval batch size
            shuffle=False
        )
        loaders.append(test_loader)
    
    return tuple(loaders)