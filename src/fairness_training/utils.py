"""
Utility functions for fairness evaluation and data processing.
"""

import numpy as np
from typing import Tuple, Dict, Optional

def compute_demographic_parity(
    predictions: np.ndarray,
    protected_attr: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute demographic parity metrics.
    
    Demographic parity is satisfied when the selection rate is equal across groups:
    P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
    
    Args:
        predictions: Predicted probabilities or scores
        protected_attr: Binary protected attribute (0 or 1)
        threshold: Decision threshold for binary predictions
        
    Returns:
        Dictionary with demographic parity metrics
    """
    binary_preds = (predictions >= threshold).astype(int).flatten()
    protected = protected_attr.flatten()
    
    # Selection rates per group
    mask_0 = protected == 0
    mask_1 = protected == 1
    
    rate_0 = binary_preds[mask_0].mean() if mask_0.sum() > 0 else 0.0
    rate_1 = binary_preds[mask_1].mean() if mask_1.sum() > 0 else 0.0
    
    # Demographic parity difference
    dp_diff = abs(rate_0 - rate_1)

    return {
        'selection_rate_group_0': rate_0,
        'selection_rate_group_1': rate_1,
        'demographic_parity_diff': dp_diff
    }


def compute_equalized_residuals(
    predictions: np.ndarray,
    targets: np.ndarray,
    protected_attr: np.ndarray
) -> Dict[str, float]:
    """
    Compute equalized residuals metrics for regression fairness.
    
    Equalized residuals is satisfied when prediction errors have the same
    distribution across protected groups:
    - E[Y - Ŷ | A=0] = E[Y - Ŷ | A=1] (mean residual equality)
    
    This ensures the model doesn't systematically over or under-predict
    for any protected group.
    
    Args:
        predictions: Predicted values (continuous)
        targets: True target values (continuous)
        protected_attr: Binary protected attribute (0 or 1)
        
    Returns:
        Dictionary with equalized residuals metrics
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    protected = protected_attr.flatten()
    
    # Compute residuals (errors)
    residuals = targets - predictions
    
    # Split by protected group
    mask_0 = protected == 0
    mask_1 = protected == 1
    
    residuals_0 = residuals[mask_0]
    residuals_1 = residuals[mask_1]
    
    # Mean residuals per group (systematic bias)
    mean_residual_0 = residuals_0.mean() if len(residuals_0) > 0 else 0.0
    mean_residual_1 = residuals_1.mean() if len(residuals_1) > 0 else 0.0

    # Fairness metrics: differences between groups
    mean_residual_diff = abs(mean_residual_0 - mean_residual_1)
    
    return {
        # Per-group statistics
        'mean_residual_group_0': mean_residual_0,
        'mean_residual_group_1': mean_residual_1,
        'mean_residual_diff': mean_residual_diff
    }

def compute_equalized_odds(
    predictions: np.ndarray,
    targets: np.ndarray,
    protected_attr: np.ndarray
) -> Dict[str, float]:
    """
    Compute equalized odds metrics based on expected values.
    
    Equalized odds is satisfied when the average predicted value is equal 
    across protected groups, separately for each true outcome:
    E[Ŷ | Y=0, A=0] = E[Ŷ | Y=0, A=1]  (for negative class)
    E[Ŷ | Y=1, A=0] = E[Ŷ | Y=1, A=1]  (for positive class)
    
    Args:
        predictions: Predicted probabilities or scores (continuous)
        targets: True labels (binary: 0 or 1)
        protected_attr: Binary protected attribute (0 or 1)
        
    Returns:
        Dictionary with equalized odds metrics:
        - Mean predictions per group per outcome
        - Absolute differences (fairness gaps)
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    protected = protected_attr.flatten()
    
    # Binarize targets if needed
    targets = (targets > 0.5).astype(int)
    
    metrics = {}
    
    # Compute mean predictions for each (protected group, outcome) combination
    for y_class in [0, 1]:
        # Mask for this outcome class
        outcome_mask = targets == y_class
        
        for group in [0, 1]:
            # Mask for this protected group
            group_mask = protected == group
            
            # Combined mask
            combined_mask = outcome_mask & group_mask
            
            if combined_mask.sum() == 0:
                # No samples in this combination
                metrics[f'mean_pred_y{y_class}_group{group}'] = np.nan
                continue
            
            # Average prediction for this (outcome, group) combination
            mean_pred = predictions[combined_mask].mean()
            metrics[f'mean_pred_y{y_class}_group{group}'] = mean_pred
        
        # Compute fairness gap for this outcome class
        key_0 = f'mean_pred_y{y_class}_group0'
        key_1 = f'mean_pred_y{y_class}_group1'
        
        if key_0 in metrics and key_1 in metrics:
            if not (np.isnan(metrics[key_0]) or np.isnan(metrics[key_1])):
                diff = abs(metrics[key_0] - metrics[key_1])
                metrics[f'eq_odds_diff_y{y_class}'] = diff
            else:
                metrics[f'eq_odds_diff_y{y_class}'] = np.nan
    
    # Overall equalized odds violation (maximum gap across both outcome classes)
    gaps = []
    for y_class in [0, 1]:
        gap_key = f'eq_odds_diff_y{y_class}'
        if gap_key in metrics and not np.isnan(metrics[gap_key]):
            gaps.append(metrics[gap_key])
    
    if gaps:
        metrics['equalized_odds_diff'] = max(gaps)
    else:
        metrics['equalized_odds_diff'] = np.nan
    
    return metrics


def prepare_data_with_protected_attr(
    X: np.ndarray,
    y: np.ndarray,
    protected_col: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: Optional[int] = None
) -> Tuple:
    """
    Split data ensuring protected attribute is binary.
    
    Args:
        X: Feature matrix
        y: Target vector
        protected_col: Index of protected attribute column
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        seed: Random seed
        
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Verify protected attribute is binary
    unique_vals = np.unique(X[:, protected_col])
    if not np.array_equal(unique_vals, [0, 1]):
        raise ValueError(
            f"Protected attribute must be binary (0/1). Found values: {unique_vals}"
        )
    
    # Shuffle indices
    n = len(X)
    indices = np.random.permutation(n)
    
    # Split indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Verify both groups present in each split
    for name, data in [('train', X_train), ('val', X_val), ('test', X_test)]:
        groups = np.unique(data[:, protected_col])
        if len(groups) < 2:
            print(f"Warning: {name} set only has group(s): {groups}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
