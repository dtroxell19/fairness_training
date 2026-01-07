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
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, Tuple, Union, List
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
        Train the model.
        
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
            
            # Store learning rate
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
            
            # Skip if missing a group in ANY protected attribute
            skip_batch = False
            for attr_idx in self.model.protected_attr_idx:
                if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                   (batch_x[:, attr_idx] == 1).sum() < 1:
                    skip_batch = True
                    break
            
            if skip_batch:
                continue
            
            # Forward pass (training mode - uses hard constraints)
            self.optimizer.zero_grad()
            
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
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Collect for aggregate gap computation
            all_preds.append(predictions.detach().cpu())
            all_targets.append(batch_y.detach().cpu())
            for attr_idx in self.model.protected_attr_idx:
                all_protected[attr_idx].append(batch_x[:, attr_idx].detach().cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute aggregate gap over ALL examples
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
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
                
                # Collect for aggregate gap computation
                all_preds.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
                for attr_idx in self.model.protected_attr_idx:
                    all_protected[attr_idx].append(batch_x[:, attr_idx].cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute aggregate gap over ALL examples
        aggregate_gap = self._compute_aggregate_gap(all_preds, all_targets, all_protected)
        
        return avg_loss, aggregate_gap
    
    def _compute_aggregate_gap(
        self,
        all_preds: list,
        all_targets: list,
        all_protected: Dict[int, list]
    ) -> float:
        """
        Compute fairness gap over ALL examples (not averaged per-batch).
        
        Args:
            all_preds: List of prediction tensors from each batch
            all_targets: List of target tensors from each batch
            all_protected: Dict mapping attr_idx -> list of protected attribute tensors
            
        Returns:
            Max fairness gap across all protected attributes
        """
        if len(all_preds) == 0:
            return 0.0
        
        # Concatenate all predictions and targets
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Concatenate protected attributes
        protected = {
            attr_idx: torch.cat(tensors, dim=0) 
            for attr_idx, tensors in all_protected.items()
        }
        
        max_gap = 0.0
        
        for attr_idx in self.model.protected_attr_idx:
            prot_vals = protected[attr_idx]
            mask_0 = prot_vals == 0
            mask_1 = prot_vals == 1
            
            if mask_0.sum() == 0 or mask_1.sum() == 0:
                continue
            
            preds_0 = preds[mask_0]
            preds_1 = preds[mask_1]
            
            if hasattr(self.model, 'fairness_criterion') and \
               self.model.fairness_criterion == 'mean_residual':
                targets_squeezed = targets.squeeze(-1) if targets.dim() > 1 else targets
                residual_0 = (targets_squeezed[mask_0] - preds_0.squeeze(-1)).mean().item()
                residual_1 = (targets_squeezed[mask_1] - preds_1.squeeze(-1)).mean().item()
                gap = max(abs(residual_0), abs(residual_1))
            else:
                mean_0 = preds_0.mean().item()
                mean_1 = preds_1.mean().item()
                gap = abs(mean_0 - mean_1)
            
            max_gap = max(max_gap, gap)
        
        return max_gap
    
    def _compute_batch_gap(
        self,
        x: torch.Tensor,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> float:
        """Compute fairness gap for a batch."""
        max_gap = 0.0
        
        for attr_idx in self.model.protected_attr_idx:
            indicator_0 = x[:, attr_idx] == 0
            indicator_1 = ~indicator_0
            
            if indicator_0.sum() == 0 or indicator_1.sum() == 0:
                continue
            
            preds_0 = predictions[indicator_0]
            preds_1 = predictions[indicator_1]
            
            if hasattr(self.model, 'fairness_criterion') and \
               self.model.fairness_criterion == 'mean_residual' and targets is not None:
                targets_squeezed = targets.squeeze(-1) if targets.dim() > 1 else targets
                residual_0 = (targets_squeezed[indicator_0] - preds_0.squeeze(-1)).mean().item()
                residual_1 = (targets_squeezed[indicator_1] - preds_1.squeeze(-1)).mean().item()
                gap = max(abs(residual_0), abs(residual_1))
            else:
                mean_0 = preds_0.mean().item()
                mean_1 = preds_1.mean().item()
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
        
        Resets inference state before evaluation, then runs inference
        on all batches (using primal-dual for small batches if applicable).
        
        Args:
            test_loader: Test data loader
            return_predictions: If True, return predictions array
            
        Returns:
            Dictionary with test metrics including per-attribute fairness gaps
        """
        print("Resetting inference state for test evaluation")
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
        
        # Add inference statistics (for small-batch regime)
        metrics['lambda_max'] = self.model.lambda_max
        metrics['total_inference_samples'] = self.model.cumulative_samples
        
        # Theoretical bound from Theorem 2.2
        if self.model.cumulative_samples > 0 and self.model.eta_0 > 0:
            eta_final = self.model.get_adaptive_eta()
            metrics['theoretical_bound'] = self.model.fairness_tolerance + \
                (self.model.lambda_max / (eta_final * self.model.cumulative_samples))
        
        if return_predictions:
            metrics['predictions'] = all_preds.numpy()
            metrics['targets'] = all_targets.numpy()
            metrics['protected'] = {
                attr_idx: all_protected[attr_idx].numpy() 
                for attr_idx in self.model.protected_attr_idx
            }
        
        return metrics
    
    def save_checkpoint(self, filepath: str, include_history: bool = True):
        """Save model checkpoint."""
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
        batch_size_train: Batch size for training (should be >= b_tau for hard constraints)
        batch_size_eval: Batch size for validation and test (if None, uses batch_size_train)
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of DataLoaders (train, val, test) - only created if data provided
    """
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
    
    # Validation loader
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        if y_val_t.dim() == 1:
            y_val_t = y_val_t.unsqueeze(1)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size_eval,
            shuffle=False,
            drop_last=True
        )
        loaders.append(val_loader)
    
    # Test loader
    if X_test is not None and y_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        if y_test_t.dim() == 1:
            y_test_t = y_test_t.unsqueeze(1)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size_eval,
            shuffle=False
        )
        loaders.append(test_loader)
    
    return tuple(loaders)


def create_stratified_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    protected_attr_idx: Union[int, List[int]] = 0,
    batch_size_train: int = 2000,
    batch_size_eval: int = 2000
) -> Tuple[DataLoader, ...]:
    """
    Create dataloaders with stratified sampling to maintain constant group ratios.
    
    Supports stratification on 1 or 2 protected attributes. With 2 attributes,
    maintains proportions for all 4 intersectional groups (00, 01, 10, 11).
    
    Samples that don't fit into complete batches are dropped with a warning.
    
    This is recommended for training (large-batch regime) to satisfy the conditions 
    of Lemma 2.1 for aggregate fairness.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)  
        X_test, y_test: Test data (optional)
        protected_attr_idx: Index or list of indices (max 2) of protected attributes
        batch_size_train: Batch size for training (should be >= b_tau)
        batch_size_eval: Batch size for validation/test
        
    Returns:
        Tuple of DataLoaders with stratified batches
        
    Example:
        # Single attribute stratification
        loaders = create_stratified_dataloaders(X, y, protected_attr_idx=0)
        
        # Two attribute stratification (maintains all 4 group proportions)
        loaders = create_stratified_dataloaders(X, y, protected_attr_idx=[0, 1])
    """
    # Normalize protected_attr_idx to list
    if isinstance(protected_attr_idx, int):
        attr_indices = [protected_attr_idx]
    else:
        attr_indices = list(protected_attr_idx)
    
    if len(attr_indices) > 2:
        raise ValueError("Stratification supports at most 2 protected attributes")
    
    def create_stratified_batches_single(X, attr_idx, batch_size, split_name):
        """Stratify on a single protected attribute (2 groups)."""
        protected = X[:, attr_idx]
        
        idx_0 = np.where(protected == 0)[0]
        idx_1 = np.where(protected == 1)[0]
        
        total = len(idx_0) + len(idx_1)
        ratio_0 = len(idx_0) / total
        
        # Calculate per-batch counts
        n_0_per_batch = int(batch_size * ratio_0)
        n_1_per_batch = batch_size - n_0_per_batch
        
        # Ensure at least 1 from each group
        if n_0_per_batch < 1 or n_1_per_batch < 1:
            raise ValueError(f"Batch size {batch_size} too small for group proportions")
        
        # Shuffle indices
        np.random.shuffle(idx_0)
        np.random.shuffle(idx_1)
        
        # Calculate how many complete batches we can make
        num_batches_0 = len(idx_0) // n_0_per_batch
        num_batches_1 = len(idx_1) // n_1_per_batch
        num_batches = min(num_batches_0, num_batches_1)
        
        # Calculate dropped samples
        used_0 = num_batches * n_0_per_batch
        used_1 = num_batches * n_1_per_batch
        dropped_0 = len(idx_0) - used_0
        dropped_1 = len(idx_1) - used_1
        total_dropped = dropped_0 + dropped_1
        
        if total_dropped > 0:
            print(f"  ⚠️  Dropping {total_dropped} samples ({dropped_0} from group 0, {dropped_1} from group 1) "
                  f"to ensure complete batches")
        
        # Create batches
        batches = []
        for b in range(num_batches):
            batch_idx_0 = idx_0[b * n_0_per_batch : (b + 1) * n_0_per_batch]
            batch_idx_1 = idx_1[b * n_1_per_batch : (b + 1) * n_1_per_batch]
            
            batch_indices = np.concatenate([batch_idx_0, batch_idx_1])
            np.random.shuffle(batch_indices)
            batches.append(batch_indices)
        
        group_counts = {0: n_0_per_batch, 1: n_1_per_batch}
        return batches, group_counts
    
    def create_stratified_batches_dual(X, attr_idx_1, attr_idx_2, batch_size, split_name):
        """Stratify on two protected attributes (4 intersectional groups)."""
        prot_1 = X[:, attr_idx_1]
        prot_2 = X[:, attr_idx_2]
        
        # Find indices for all 4 intersectional groups
        idx_00 = np.where((prot_1 == 0) & (prot_2 == 0))[0]
        idx_01 = np.where((prot_1 == 0) & (prot_2 == 1))[0]
        idx_10 = np.where((prot_1 == 1) & (prot_2 == 0))[0]
        idx_11 = np.where((prot_1 == 1) & (prot_2 == 1))[0]
        
        groups = {'00': idx_00, '01': idx_01, '10': idx_10, '11': idx_11}
        total = sum(len(idx) for idx in groups.values())
        
        # Calculate per-batch counts for each group
        counts_per_batch = {}
        remaining = batch_size
        
        for i, (name, idx) in enumerate(groups.items()):
            ratio = len(idx) / total
            if i < 3:  # First 3 groups: use floor
                count = int(batch_size * ratio)
                counts_per_batch[name] = max(1, count)  # Ensure at least 1
                remaining -= counts_per_batch[name]
            else:  # Last group: take remainder to ensure exact batch size
                counts_per_batch[name] = max(1, remaining)
        
        # Verify batch size
        actual_batch = sum(counts_per_batch.values())
        if actual_batch != batch_size:
            # Adjust the largest group
            diff = batch_size - actual_batch
            largest_group = max(counts_per_batch, key=counts_per_batch.get)
            counts_per_batch[largest_group] += diff
        
        # Shuffle all group indices
        for name in groups:
            np.random.shuffle(groups[name])
        
        # Calculate how many complete batches we can make
        num_batches_per_group = {
            name: len(groups[name]) // counts_per_batch[name] 
            for name in groups
        }
        num_batches = min(num_batches_per_group.values())
        
        # Calculate dropped samples
        total_dropped = 0
        dropped_per_group = {}
        for name in groups:
            used = num_batches * counts_per_batch[name]
            dropped = len(groups[name]) - used
            dropped_per_group[name] = dropped
            total_dropped += dropped
        
        if total_dropped > 0:
            dropped_str = ", ".join([f"group {k}: {v}" for k, v in dropped_per_group.items() if v > 0])
            print(f"  ⚠️  Dropping {total_dropped} samples ({dropped_str}) to ensure complete batches")
        
        # Create batches
        batches = []
        for b in range(num_batches):
            batch_indices_list = []
            for name in groups:
                start = b * counts_per_batch[name]
                end = (b + 1) * counts_per_batch[name]
                batch_indices_list.append(groups[name][start:end])
            
            batch_indices = np.concatenate(batch_indices_list)
            np.random.shuffle(batch_indices)
            batches.append(batch_indices)
        
        return batches, counts_per_batch
    
    def create_loader(X, y, batch_size, split_name):
        """Create a single stratified dataloader."""
        print(f"\n=== Creating stratified {split_name} batches ===")
        print(f"  Original samples: {len(X)}")
        
        if len(attr_indices) == 1:
            batches, group_counts = create_stratified_batches_single(
                X, attr_indices[0], batch_size, split_name
            )
            print(f"  Per-batch allocation: group 0={group_counts[0]}, group 1={group_counts[1]}")
        else:
            batches, group_counts = create_stratified_batches_dual(
                X, attr_indices[0], attr_indices[1], batch_size, split_name
            )
            counts_str = ", ".join([f"group {k}={v}" for k, v in group_counts.items()])
            print(f"  Per-batch allocation: {counts_str}")
        
        print(f"  Created {len(batches)} batches of size {batch_size}")
        print(f"  Total samples used: {len(batches) * batch_size}")
        
        if len(batches) == 0:
            raise ValueError(f"No complete batches could be created for {split_name}. "
                           f"Try reducing batch_size or adding more data.")
        
        # Reorder data according to batches
        all_indices = np.concatenate(batches)
        X_ordered = torch.tensor(X[all_indices], dtype=torch.float32)
        y_ordered = torch.tensor(y[all_indices], dtype=torch.float32)
        if y_ordered.dim() == 1:
            y_ordered = y_ordered.unsqueeze(1)
        
        dataset = TensorDataset(X_ordered, y_ordered)
        
        # Simple sequential batching since data is already ordered
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Data already arranged in stratified batches
            drop_last=False  # We've already ensured complete batches
        )
        
        return loader
    
    loaders = []
    
    # Training loader
    train_loader = create_loader(X_train, y_train, batch_size_train, "training")
    loaders.append(train_loader)
    
    # Validation loader
    if X_val is not None and y_val is not None:
        val_loader = create_loader(X_val, y_val, batch_size_eval, "validation")
        loaders.append(val_loader)
    
    # Test loader
    if X_test is not None and y_test is not None:
        test_loader = create_loader(X_test, y_test, batch_size_eval, "test")
        loaders.append(test_loader)
    
    return tuple(loaders)

