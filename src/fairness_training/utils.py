"""
Utility functions for fairness evaluation
"""

import logging
import numpy as np
import torch
import cvxpy as cp
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Union, List
from .fairness_metrics import FairnessMetric, MeanPredictionParity, MeanResidualFairness, EqualizedOdds

log = logging.getLogger(__name__)

def get_fairness_metric(
    metric: Union[str, FairnessMetric],
    num_protected_attrs: int = 1
) -> FairnessMetric:
    """
    Get a fairness metric instance from a string name or return the metric if already instantiated.
    
    Args:
        metric: Either a string ('mean_pred', 'mean_residual', 'equalized_odds') 
                or a FairnessMetric instance
        num_protected_attrs: Number of protected attributes (used if creating from string)
        
    Returns:
        FairnessMetric instance
    """
    if isinstance(metric, FairnessMetric):
        return metric
    
    if isinstance(metric, str):
        metric_map = {
            'mean_pred': MeanPredictionParity,
            'mean_prediction_parity': MeanPredictionParity,
            'demographic_parity': MeanPredictionParity,
            'mean_residual': MeanResidualFairness,
            'mean_residual_fairness': MeanResidualFairness,
            'equalized_residuals': MeanResidualFairness,
            'equalized_odds': EqualizedOdds,
        }
        
        metric_lower = metric.lower()
        if metric_lower not in metric_map:
            raise ValueError(
                f"Unknown fairness metric: '{metric}'. "
                f"Available options: {list(metric_map.keys())} or provide a FairnessMetric instance."
            )
        
        return metric_map[metric_lower](num_protected_attrs=num_protected_attrs)
    
    raise TypeError(
        f"metric must be a string or FairnessMetric instance, got {type(metric)}"
    )

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
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor],
    X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
    y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
    protected_attr_idx: Union[int, List[int]] = 0,
    batch_size_train: int = 256,
    batch_size_eval: int = 256,
) -> Tuple[DataLoader, ...]:
    """
    Create dataloaders with stratified sampling to maintain constant group ratios.
    
    Supports stratification on 1 or 2 protected attributes. With 2 attributes,
    maintains proportions for all 4 intersectional groups (00, 01, 10, 11).
    
    Samples that don't fit into complete batches are dropped with a warning.
    
    This is recommended for training.
    
    Args:
        X_train, y_train: Training data (numpy array or torch.Tensor).
        X_val, y_val: Validation data (optional).
        X_test, y_test: Test data (optional).
        protected_attr_idx: Index or list of indices (max 2) of protected attributes.
        batch_size_train: Batch size for training.
        batch_size_eval: Batch size for validation/test.
        
    Returns:
        Tuple of DataLoaders with stratified batches
        
    Example:
        # Single attribute stratification
        loaders = create_stratified_dataloaders(X, y, protected_attr_idx=0)
        
        # Two attribute stratification (maintains all 4 group proportions)
        loaders = create_stratified_dataloaders(X, y, protected_attr_idx=[0, 1])
    """
    # Accept torch.Tensor as well as numpy arrays
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    X_train = _to_numpy(X_train)
    y_train = _to_numpy(y_train)
    if X_val is not None:
        X_val = _to_numpy(X_val)
    if y_val is not None:
        y_val = _to_numpy(y_val)
    if X_test is not None:
        X_test = _to_numpy(X_test)
    if y_test is not None:
        y_test = _to_numpy(y_test)

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
        
        # Validate that attribute is binary
        unique_vals = np.unique(protected)
        if not (set(unique_vals) <= {0, 1}):
            raise ValueError(
                f"Protected attribute at index {attr_idx} is not binary (0/1). "
                f"Found unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}. "
                f"Ensure your protected attribute column contains only 0 and 1 values."
            )
        
        idx_0 = np.where(protected == 0)[0]
        idx_1 = np.where(protected == 1)[0]
        
        if len(idx_0) == 0 or len(idx_1) == 0:
            raise ValueError(
                f"Protected attribute at index {attr_idx} must have samples in both groups. "
                f"Found: group 0 = {len(idx_0)} samples, group 1 = {len(idx_1)} samples."
            )
        
        total = len(idx_0) + len(idx_1)
        ratio_0 = len(idx_0) / total
        

        n_0_per_batch = int(batch_size * ratio_0)
        n_1_per_batch = batch_size - n_0_per_batch
        if n_0_per_batch < 1 or n_1_per_batch < 1:
            raise ValueError(f"Batch size {batch_size} too small for group proportions")
        
        np.random.shuffle(idx_0)
        np.random.shuffle(idx_1)
        
        # Calculate how many complete batches we can make + how many dropped
        num_batches_0 = len(idx_0) // n_0_per_batch
        num_batches_1 = len(idx_1) // n_1_per_batch
        num_batches = min(num_batches_0, num_batches_1)
        
        used_0 = num_batches * n_0_per_batch
        used_1 = num_batches * n_1_per_batch
        dropped_0 = len(idx_0) - used_0
        dropped_1 = len(idx_1) - used_1
        total_dropped = dropped_0 + dropped_1
        
        if total_dropped > 0:
            import warnings as _w
            _w.warn(
                f"create_stratified_dataloaders: dropping {total_dropped} samples "
                f"({dropped_0} from group 0, {dropped_1} from group 1) to form "
                f"complete batches.",
                UserWarning,
                stacklevel=4,
            )
        
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
        
        # Ensure binary
        unique_1 = np.unique(prot_1)
        unique_2 = np.unique(prot_2)
        
        if not (set(unique_1) <= {0, 1}):
            raise ValueError(
                f"Protected attribute at index {attr_idx_1} is not binary (0/1). "
                f"Found unique values: {unique_1[:10]}{'...' if len(unique_1) > 10 else ''}. "
                f"For continuous features, use single-attribute stratification."
            )
        if not (set(unique_2) <= {0, 1}):
            raise ValueError(
                f"Protected attribute at index {attr_idx_2} is not binary (0/1). "
                f"Found unique values: {unique_2[:10]}{'...' if len(unique_2) > 10 else ''}. "
                f"For continuous features, use single-attribute stratification."
            )
        
        # Find indices for all 4 intersectional groups
        idx_00 = np.where((prot_1 == 0) & (prot_2 == 0))[0]
        idx_01 = np.where((prot_1 == 0) & (prot_2 == 1))[0]
        idx_10 = np.where((prot_1 == 1) & (prot_2 == 0))[0]
        idx_11 = np.where((prot_1 == 1) & (prot_2 == 1))[0]
        
        groups = {'00': idx_00, '01': idx_01, '10': idx_10, '11': idx_11}
        total = sum(len(idx) for idx in groups.values())
        
        if total == 0:
            raise ValueError(
                f"No samples found in any intersectional group. "
                f"Check that columns {attr_idx_1} and {attr_idx_2} contain binary (0/1) values."
            )
        
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
        
        for name in groups:
            np.random.shuffle(groups[name])
        
        # Calculate how many complete batches we can make + # dropped samples
        num_batches_per_group = {
            name: len(groups[name]) // counts_per_batch[name] 
            for name in groups
        }
        num_batches = min(num_batches_per_group.values())
        
        total_dropped = 0
        dropped_per_group = {}
        for name in groups:
            used = num_batches * counts_per_batch[name]
            dropped = len(groups[name]) - used
            dropped_per_group[name] = dropped
            total_dropped += dropped
        
        if total_dropped > 0:
            dropped_str = ", ".join([f"group {k}: {v}" for k, v in dropped_per_group.items() if v > 0])
            import warnings as _w
            _w.warn(
                f"create_stratified_dataloaders: dropping {total_dropped} samples "
                f"({dropped_str}) to form complete batches.",
                UserWarning,
                stacklevel=4,
            )
        
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
        log.debug("Creating stratified %s batches (n=%d)", split_name, len(X))

        if len(attr_indices) == 1:
            batches, group_counts = create_stratified_batches_single(
                X, attr_indices[0], batch_size, split_name
            )
            log.debug("  per-batch: group 0=%d, group 1=%d", group_counts[0], group_counts[1])
        else:
            batches, group_counts = create_stratified_batches_dual(
                X, attr_indices[0], attr_indices[1], batch_size, split_name
            )
            counts_str = ", ".join([f"group {k}={v}" for k, v in group_counts.items()])
            log.debug("  per-batch: %s", counts_str)

        log.debug("  %d batches of size %d (%d total samples)",
                  len(batches), batch_size, len(batches) * batch_size)
        
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
    
    train_loader = create_loader(X_train, y_train, batch_size_train, "training")
    loaders.append(train_loader)
    
    if X_val is not None and y_val is not None:
        val_loader = create_loader(X_val, y_val, batch_size_eval, "validation")
        loaders.append(val_loader)
    
    if X_test is not None and y_test is not None:
        test_loader = create_loader(X_test, y_test, batch_size_eval, "test")
        loaders.append(test_loader)

    return tuple(loaders)


def validate_metric(
    metric: FairnessMetric,
    input_dim: int,
    batch_size: int = 20,
    protected_attr_idx: Union[int, List[int]] = 0,
    fairness_tolerance: float = 0.05,
) -> None:
    """Validate that a FairnessMetric produces DPP-compliant cvxpy constraints.

    Runs a dry-pass through ``metric.create_constraints()`` using dummy cvxpy
    objects and synthetic data, then checks whether the resulting problem is
    DPP-compliant (``cp.Problem.is_dcp(dpp=True)``).  Raises ``ValueError``
    with a descriptive message if any constraint fails.

    Call this before training to catch DPP violations early, rather than
    receiving a cryptic solver error on the first batch.

    Args:
        metric: A :class:`FairnessMetric` instance to validate.
        input_dim: Number of input features (columns) in X.
        batch_size: Number of samples to use in the synthetic batch.
        protected_attr_idx: Column index or list of indices of protected attrs.
        fairness_tolerance: Epsilon value for the slack parameter.

    Raises:
        TypeError: If ``metric`` is not a ``FairnessMetric`` instance.
        ValueError: If any constraint is not DPP-compliant, with the index of
                    the offending constraint and the expression string.

    Example::

        from fairness_training import validate_metric, MeanPredictionParity

        validate_metric(MeanPredictionParity(), input_dim=20, batch_size=100)
        # Passes silently if valid.
    """
    if not isinstance(metric, FairnessMetric):
        raise TypeError(
            f"validate_metric expects a FairnessMetric instance, got {type(metric).__name__}. "
            f"Make sure you are passing an instantiated metric object (e.g. "
            f"MeanPredictionParity()), not a class or string."
        )

    if isinstance(protected_attr_idx, int):
        attr_indices = [protected_attr_idx]
    else:
        attr_indices = list(protected_attr_idx)

    # Build synthetic torch tensors for selection matrix construction
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    # Ensure both groups present for every protected attr
    for idx in attr_indices:
        x_np[:, idx] = 0.0
        x_np[batch_size // 2:, idx] = 1.0

    y_np = (rng.random(batch_size) > 0.5).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np).unsqueeze(1)

    selection_matrices = metric.create_selection_matrices(x_t, y_t, attr_indices)

    # Build dummy cvxpy objects
    yhat = cp.Variable(batch_size)
    slack = cp.Parameter(1, nonneg=True)
    slack.value = np.array([fairness_tolerance])

    if metric.requires_y_in_constraints:
        y_param = cp.Parameter(batch_size)
        y_param.value = y_np
        constraints = metric.create_constraints(yhat, selection_matrices, slack, y_param)
    else:
        constraints = metric.create_constraints(yhat, selection_matrices, slack)

    if not constraints:
        return  # No constraints — nothing to validate

    objective = cp.Minimize(cp.sum_squares(yhat))
    problem = cp.Problem(objective, constraints)

    if not problem.is_dcp(dpp=True):
        # Find the first offending constraint
        bad_indices = []
        for i, c in enumerate(constraints):
            p = cp.Problem(cp.Minimize(0), [c])
            if not p.is_dcp(dpp=True):
                bad_indices.append((i, str(c)))

        detail = "; ".join(f"constraint[{i}]: {expr}" for i, expr in bad_indices[:3])
        raise ValueError(
            f"validate_metric: {type(metric).__name__} produces constraints that are not "
            f"DPP-compliant. DPP (Disciplined Parameterized Programming) requires constraints "
            f"to be affine in decision variables and parameters separately — avoid multiplying "
            f"two cvxpy Parameters or a Parameter by a Variable.\n"
            f"Offending constraint(s): {detail}\n"
            f"See docs/user-guide/custom-metrics.md for DPP compliance rules."
        )
