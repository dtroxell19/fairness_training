"""
Utility functions for fairness evaluation and data processing.

This module provides:
- Fairness metric computation (demographic parity, equalized odds, equalized residuals)
- Data preprocessing utilities
- Stratified data splitting for fairness experiments
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Union


# =============================================================================
# FAIRNESS METRICS
# =============================================================================

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
    
    # Demographic parity ratio (min/max)
    if rate_0 > 0 and rate_1 > 0:
        dp_ratio = min(rate_0, rate_1) / max(rate_0, rate_1)
    else:
        dp_ratio = 0.0

    return {
        'selection_rate_group_0': rate_0,
        'selection_rate_group_1': rate_1,
        'demographic_parity_diff': dp_diff,
        'demographic_parity_ratio': dp_ratio
    }


def compute_mean_prediction_parity(
    predictions: np.ndarray,
    protected_attr: np.ndarray
) -> Dict[str, float]:
    """
    Compute mean prediction parity (expected conditional parity).
    
    This is the fairness criterion used in the paper for the large-batch regime:
    |E[Ŷ | A=0] - E[Ŷ | A=1]| ≤ ε
    
    Args:
        predictions: Predicted values (continuous or probabilities)
        protected_attr: Binary protected attribute (0 or 1)
        
    Returns:
        Dictionary with mean prediction parity metrics
    """
    predictions = predictions.flatten()
    protected = protected_attr.flatten()
    
    mask_0 = protected == 0
    mask_1 = protected == 1
    
    mean_0 = predictions[mask_0].mean() if mask_0.sum() > 0 else 0.0
    mean_1 = predictions[mask_1].mean() if mask_1.sum() > 0 else 0.0
    
    std_0 = predictions[mask_0].std() if mask_0.sum() > 0 else 0.0
    std_1 = predictions[mask_1].std() if mask_1.sum() > 0 else 0.0
    
    gap = abs(mean_0 - mean_1)
    
    return {
        'mean_pred_group_0': mean_0,
        'mean_pred_group_1': mean_1,
        'std_pred_group_0': std_0,
        'std_pred_group_1': std_1,
        'mean_prediction_gap': gap,
        'n_group_0': int(mask_0.sum()),
        'n_group_1': int(mask_1.sum())
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
    - E[Y - Ŷ | A=0] ≈ 0  (mean residual for group 0)
    - E[Y - Ŷ | A=1] ≈ 0  (mean residual for group 1)
    
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

    # Fairness metrics
    mean_residual_diff = abs(mean_residual_0 - mean_residual_1)
    
    return {
        # Per-group statistics
        'mean_residual_group_0': mean_residual_0,
        'mean_residual_group_1': mean_residual_1,
        # Fairness metrics
        'mean_residual_diff': mean_residual_diff,
        'n_group_0': int(mask_0.sum()),
        'n_group_1': int(mask_1.sum())
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
                metrics[f'n_y{y_class}_group{group}'] = 0
                continue
            
            # Average prediction for this (outcome, group) combination
            mean_pred = predictions[combined_mask].mean()
            metrics[f'mean_pred_y{y_class}_group{group}'] = mean_pred
            metrics[f'n_y{y_class}_group{group}'] = int(combined_mask.sum())
        
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


def compute_all_fairness_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    protected_attr: np.ndarray,
    threshold: float = 0.5,
    is_classification: bool = True
) -> Dict[str, float]:
    """
    Compute all fairness metrics in one call.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        protected_attr: Binary protected attribute
        threshold: Decision threshold for classification
        is_classification: Whether this is a classification task
        
    Returns:
        Combined dictionary with all fairness metrics
    """
    metrics = {}
    
    # Mean prediction parity (always applicable)
    mp_metrics = compute_mean_prediction_parity(predictions, protected_attr)
    metrics.update({f'mp_{k}': v for k, v in mp_metrics.items()})
    
    if is_classification:
        # Demographic parity
        dp_metrics = compute_demographic_parity(predictions, protected_attr, threshold)
        metrics.update({f'dp_{k}': v for k, v in dp_metrics.items()})
        
        # Equalized odds
        eo_metrics = compute_equalized_odds(predictions, targets, protected_attr)
        metrics.update({f'eo_{k}': v for k, v in eo_metrics.items()})
    else:
        # Equalized residuals (for regression)
        er_metrics = compute_equalized_residuals(predictions, targets, protected_attr)
        metrics.update({f'er_{k}': v for k, v in er_metrics.items()})
    
    return metrics


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

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


def binarize_protected_attribute(
    data: np.ndarray,
    column: int,
    positive_values: Union[List, np.ndarray]
) -> np.ndarray:
    """
    Convert a categorical protected attribute to binary.
    
    Args:
        data: Feature matrix
        column: Index of column to binarize
        positive_values: Values that should be mapped to 1
        
    Returns:
        Modified feature matrix with binary protected attribute
    """
    data = data.copy()
    binary_col = np.isin(data[:, column], positive_values).astype(float)
    data[:, column] = binary_col
    return data


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_fairness_data(
    n_samples: int = 10000,
    n_features: int = 20,
    protected_ratio: float = 0.3,
    group_bias: float = 2.0,
    noise_std: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for fairness experiments.
    
    Creates a dataset where:
    - First column is binary protected attribute
    - Some features are correlated with protected attribute
    - Target has systematic bias based on protected attribute
    
    Args:
        n_samples: Number of samples
        n_features: Total number of features (including protected)
        protected_ratio: Proportion in protected group (A=1)
        group_bias: Bias added to target for protected group
        noise_std: Standard deviation of noise
        seed: Random seed
        
    Returns:
        X: Feature matrix with protected attribute in column 0
        y: Target vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Protected attribute (binary)
    protected = np.random.binomial(1, protected_ratio, n_samples)
    
    # Features (some correlated with protected attribute)
    X_features = np.random.randn(n_samples, n_features - 1)
    
    # Add correlation with protected attribute for some features
    for i in range(min(5, n_features - 1)):
        X_features[:, i] += 0.3 * protected * (i + 1)
    
    # Combine: protected attribute as first column
    X = np.hstack([protected.reshape(-1, 1), X_features]).astype(np.float32)
    
    # Generate target with coefficients and group bias
    coefficients = np.random.randn(n_features - 1)
    coefficients[:5] = [3.0, -2.0, 1.5, -1.0, 0.5]  # Known coefficients for first features
    
    y_true = X_features @ coefficients + np.random.randn(n_samples) * noise_std
    y_biased = y_true + group_bias * protected  # Add systematic bias
    y = y_biased.astype(np.float32)
    
    return X, y


def generate_synthetic_classification_data(
    n_samples: int = 10000,
    n_features: int = 20,
    protected_ratio: float = 0.3,
    base_positive_rate: float = 0.25,
    group_positive_rate_diff: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic binary classification data for fairness experiments.
    
    Creates a dataset where the positive rate differs between groups,
    which is typical in fairness-critical applications.
    
    Args:
        n_samples: Number of samples
        n_features: Total number of features (including protected)
        protected_ratio: Proportion in protected group (A=1)
        base_positive_rate: Base rate of positive class
        group_positive_rate_diff: Difference in positive rate for protected group
        seed: Random seed
        
    Returns:
        X: Feature matrix with protected attribute in column 0
        y: Binary target vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Protected attribute (binary)
    protected = np.random.binomial(1, protected_ratio, n_samples)
    
    # Features
    X_features = np.random.randn(n_samples, n_features - 1)
    
    # Add some correlation with protected attribute
    for i in range(min(3, n_features - 1)):
        X_features[:, i] += 0.5 * protected
    
    # Combine: protected attribute as first column
    X = np.hstack([protected.reshape(-1, 1), X_features]).astype(np.float32)
    
    # Generate latent score
    coefficients = np.zeros(n_features - 1)
    coefficients[:5] = [1.0, -0.5, 0.8, -0.3, 0.6]
    latent_score = X_features @ coefficients
    
    # Adjust threshold for each group to create different positive rates
    group_0_threshold = np.percentile(latent_score[protected == 0], 
                                       100 * (1 - base_positive_rate))
    group_1_threshold = np.percentile(latent_score[protected == 1], 
                                       100 * (1 - base_positive_rate - group_positive_rate_diff))
    
    # Generate binary labels
    y = np.zeros(n_samples, dtype=np.float32)
    y[(protected == 0) & (latent_score >= group_0_threshold)] = 1
    y[(protected == 1) & (latent_score >= group_1_threshold)] = 1
    
    return X, y


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def print_fairness_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    protected_attr: np.ndarray,
    protected_name: str = "Protected Attribute",
    is_classification: bool = True
):
    """
    Print a comprehensive fairness report.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets  
        protected_attr: Binary protected attribute
        protected_name: Name of the protected attribute
        is_classification: Whether this is a classification task
    """
    print("\n" + "="*60)
    print(f"FAIRNESS REPORT: {protected_name}")
    print("="*60)
    
    # Group statistics
    mask_0 = protected_attr.flatten() == 0
    mask_1 = ~mask_0
    
    print(f"\nGroup Distribution:")
    print(f"  Group 0: {mask_0.sum()} samples ({100*mask_0.mean():.1f}%)")
    print(f"  Group 1: {mask_1.sum()} samples ({100*mask_1.mean():.1f}%)")
    
    # Mean prediction parity
    print(f"\nMean Prediction Parity:")
    mp = compute_mean_prediction_parity(predictions, protected_attr)
    print(f"  E[Ŷ | A=0] = {mp['mean_pred_group_0']:.4f}")
    print(f"  E[Ŷ | A=1] = {mp['mean_pred_group_1']:.4f}")
    print(f"  Gap: {mp['mean_prediction_gap']:.4f}")
    
    if is_classification:
        # Demographic parity
        print(f"\nDemographic Parity:")
        dp = compute_demographic_parity(predictions, protected_attr)
        print(f"  P(Ŷ=1 | A=0) = {dp['selection_rate_group_0']:.4f}")
        print(f"  P(Ŷ=1 | A=1) = {dp['selection_rate_group_1']:.4f}")
        print(f"  Gap: {dp['demographic_parity_diff']:.4f}")
        print(f"  Ratio: {dp['demographic_parity_ratio']:.4f}")
        
        # Equalized odds
        print(f"\nEqualized Odds:")
        eo = compute_equalized_odds(predictions, targets, protected_attr)
        print(f"  E[Ŷ | Y=0, A=0] = {eo.get('mean_pred_y0_group0', np.nan):.4f}")
        print(f"  E[Ŷ | Y=0, A=1] = {eo.get('mean_pred_y0_group1', np.nan):.4f}")
        print(f"  E[Ŷ | Y=1, A=0] = {eo.get('mean_pred_y1_group0', np.nan):.4f}")
        print(f"  E[Ŷ | Y=1, A=1] = {eo.get('mean_pred_y1_group1', np.nan):.4f}")
        print(f"  Max Gap: {eo['equalized_odds_diff']:.4f}")
    else:
        # Equalized residuals
        print(f"\nEqualized Residuals:")
        er = compute_equalized_residuals(predictions, targets, protected_attr)
        print(f"  E[Y - Ŷ | A=0] = {er['mean_residual_group_0']:.4f}")
        print(f"  E[Y - Ŷ | A=1] = {er['mean_residual_group_1']:.4f}")
        print(f"  Gap: {er['mean_residual_diff']:.4f}")
        print(f"  Max |mean residual|: {er['max_abs_mean_residual']:.4f}")
    
    print("="*60 + "\n")


def check_fairness_constraint(
    predictions: np.ndarray,
    protected_attr: np.ndarray,
    tolerance: float,
    fairness_criterion: str = 'mean_pred',
    targets: Optional[np.ndarray] = None
) -> Tuple[bool, float]:
    """
    Check if fairness constraint is satisfied.
    
    Args:
        predictions: Model predictions
        protected_attr: Binary protected attribute
        tolerance: Maximum allowed fairness gap
        fairness_criterion: 'mean_pred' or 'mean_residual'
        targets: Required if fairness_criterion='mean_residual'
        
    Returns:
        (is_satisfied, gap): Whether constraint is satisfied and the actual gap
    """
    predictions = predictions.flatten()
    protected = protected_attr.flatten()
    
    mask_0 = protected == 0
    mask_1 = ~mask_0
    
    if fairness_criterion == 'mean_residual':
        if targets is None:
            raise ValueError("targets required for mean_residual criterion")
        targets = targets.flatten()
        residual_0 = (targets[mask_0] - predictions[mask_0]).mean()
        residual_1 = (targets[mask_1] - predictions[mask_1]).mean()
        gap = max(abs(residual_0), abs(residual_1))
    else:  # mean_pred
        mean_0 = predictions[mask_0].mean() if mask_0.sum() > 0 else 0
        mean_1 = predictions[mask_1].mean() if mask_1.sum() > 0 else 0
        gap = abs(mean_0 - mean_1)
    
    is_satisfied = gap <= tolerance
    return is_satisfied, gap
