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

import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, Tuple
import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


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
        # Warn if evaluation batch size is smaller than b_tau — inference will fall
        # back to the primal-dual algorithm instead of hard per-batch constraints.
        # (Training always uses hard constraints regardless of b_tau.)
        if hasattr(self.model, 'b_tau') and val_loader is not None:
            eval_bs = val_loader.batch_size
            if eval_bs is not None and eval_bs < self.model.b_tau:
                warnings.warn(
                    f"Validation batch size ({eval_bs}) is smaller than b_tau "
                    f"({self.model.b_tau}). Inference will use the primal-dual algorithm "
                    f"rather than hard per-batch constraints, which gives aggregate — not "
                    f"per-batch — fairness guarantees. To use hard constraints at inference, "
                    f"increase the evaluation batch size or lower b_tau.",
                    UserWarning,
                    stacklevel=2,
                )

        use_tqdm = verbose >= 1 and _HAS_TQDM
        epoch_iter = _tqdm(range(epochs), desc="Training", unit="epoch") if use_tqdm else range(epochs)

        for epoch in epoch_iter:
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
                    msg = (
                        f"Epoch {epoch+1:4d} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Train Gap: {train_gap:.6f} | "
                        f"Val Gap: {val_gap:.6f}"
                    )
                    if use_tqdm:
                        epoch_iter.write(msg)
                    else:
                        print(msg)
                    if verbose >= 2:
                        lr_msg = f"           | LR: {lr:.6f}"
                        epoch_iter.write(lr_msg) if use_tqdm else print(lr_msg)

                # Early stopping
                if self.patience_counter >= self.patience:
                    stop_msg = f"\nEarly stopping at epoch {epoch + 1}"
                    if verbose >= 1:
                        epoch_iter.write(stop_msg) if use_tqdm else print(stop_msg)
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    break
            else:
                # No validation set - just log training
                if verbose >= 1 and (epoch + 1) % log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    msg = (
                        f"Epoch {epoch+1:4d} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Train Gap: {train_gap:.6f} | "
                        f"LR: {lr:.6f}"
                    )
                    epoch_iter.write(msg) if use_tqdm else print(msg)

            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch with hard fairness constraints."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0

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
                skipped_batches += 1
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
        
        if skipped_batches > 0:
            warnings.warn(
                f"Training epoch: {skipped_batches} batch(es) skipped because at least one "
                f"protected group was absent. Loss and fairness gap are averaged over the "
                f"{num_batches} non-skipped batches. Use create_stratified_dataloaders() to "
                f"prevent skipping.",
                UserWarning,
                stacklevel=3,
            )

        avg_loss = total_loss / max(num_batches, 1)
        aggregate_gap = self._compute_aggregate_gap(all_preds, all_targets, all_protected)

        return avg_loss, aggregate_gap

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0

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
                    skipped_batches += 1
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

        if skipped_batches > 0:
            warnings.warn(
                f"Validation: {skipped_batches} batch(es) skipped because at least one "
                f"protected group was absent. Metrics are computed on the remaining "
                f"{num_batches} batches only.",
                UserWarning,
                stacklevel=3,
            )

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
        skipped_batches = 0

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
                    skipped_batches += 1
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

        if skipped_batches > 0:
            warnings.warn(
                f"evaluate: {skipped_batches} batch(es) skipped because at least one "
                f"protected group was absent. Metrics are computed on the remaining "
                f"{num_batches} batches only.",
                UserWarning,
                stacklevel=2,
            )

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

        # Pooled aggregate gap — all predictions concatenated across batches.
        # Coincides with weighted_avg_fairness_gap only when group ratios are
        # constant across batches (e.g., when stratified sampling is used).
        pooled_gap = self.model.fairness_metric.compute_gap(
            all_preds.numpy(),
            all_targets.numpy(),
            protected_indicators
        )

        # Per-attribute gaps
        max_pooled_gap = 0.0
        for attr_idx in self.model.protected_attr_idx:
            protected = all_protected[attr_idx]
            mask_0 = protected == 0
            mask_1 = protected == 1

            if mask_0.sum() == 0 or mask_1.sum() == 0:
                continue

            single_attr_indicators = {attr_idx: protected.numpy()}
            attr_gap = self.model.fairness_metric.compute_gap(
                all_preds.numpy(),
                all_targets.numpy(),
                single_attr_indicators
            )
            metrics[f'fairness_gap_attr_{attr_idx}'] = attr_gap
            max_pooled_gap = max(max_pooled_gap, attr_gap)

        metrics['pooled_fairness_gap'] = max_pooled_gap

        # Batch-size-weighted time-average of per-batch gaps — the quantity bounded by
        # the theorem. Only meaningful in the primal-dual (small-batch) regime.
        if self.model.cumulative_samples > 0:
            metrics['weighted_avg_fairness_gap'] = (
                self.model.cumulative_weighted_violation / self.model.cumulative_samples
                + self.model.fairness_tolerance
            )
        else:
            # Large-batch regime: hard constraints mean pooled and weighted-avg coincide
            metrics['weighted_avg_fairness_gap'] = max_pooled_gap

        # Keep 'fairness_gap' as an alias for backwards compatibility
        metrics['fairness_gap'] = max_pooled_gap

        # Inference statistics
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
        """Save model checkpoint including fairness hyperparameters."""
        # Capture enough model config to reconstruct FairModel without guessing
        model_config = {
            'input_dim': getattr(self.model, 'input_dim', None),
            'hidden_dims': getattr(self.model, 'hidden_dims', None),
            'output_dim': getattr(self.model, 'output_dim', None),
            'protected_attr_idx': getattr(self.model, 'protected_attr_idx', None),
            'prediction_bounds': (
                getattr(self.model, 'lb', None),
                getattr(self.model, 'ub', None),
            ),
            'fairness_tolerance': getattr(self.model, 'fairness_tolerance', None),
            'b_tau': getattr(self.model, 'b_tau', None),
            'eta_0': getattr(self.model, 'eta_0', None),
            'fairness_metric': type(self.model.fairness_metric).__name__
                if hasattr(self.model, 'fairness_metric') else None,
        }

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': model_config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
        }

        if include_history:
            checkpoint['history'] = self.history

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint.

        If the checkpoint contains a 'model_config' entry, its values are compared
        against the current model's configuration and a warning is issued for any
        mismatch so that silent config drift is caught early.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Validate config if present
        saved_config = checkpoint.get('model_config', {})
        if saved_config:
            mismatches = []
            checks = {
                'protected_attr_idx': getattr(self.model, 'protected_attr_idx', None),
                'fairness_tolerance': getattr(self.model, 'fairness_tolerance', None),
                'b_tau': getattr(self.model, 'b_tau', None),
                'fairness_metric': type(self.model.fairness_metric).__name__
                    if hasattr(self.model, 'fairness_metric') else None,
            }
            for key, current_val in checks.items():
                saved_val = saved_config.get(key)
                if saved_val is not None and saved_val != current_val:
                    mismatches.append(f"  {key}: saved={saved_val!r}, current={current_val!r}")
            if mismatches:
                import warnings as _w
                _w.warn(
                    "Checkpoint config does not match current model:\n"
                    + "\n".join(mismatches)
                    + "\nEnsure you reconstructed FairModel with the same hyperparameters.",
                    UserWarning,
                    stacklevel=2,
                )

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
