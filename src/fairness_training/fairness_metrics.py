"""
Fairness Metrics for Differentiable Fairness Layers

This module provides an extensible framework for defining fairness metrics that
can be used with the FairModel class. Users can either use built-in metrics or
create custom metrics by subclassing FairnessMetric.

Requirements for Custom Metrics:
1. Must be affine (linear in predictions and/or targets)
2. Must be DPP-compliant for cvxpylayers:
   - All parameters are classified as affine (like variables)
   - Product of two expressions is affine only when:
     * At least one expression is constant, OR
     * One is parameter-affine and the other is parameter-free
   - Use selection matrices (constant 0/1 numpy arrays) for group selection
   - Avoid multiplying parameters with other parameters

Built-in Metrics:
- MeanPredictionParity: |E[Y_hat|A=0] - E[Y_hat|A=1]| <= epsilon (expected demographic parity)
- MeanResidualFairness: |E[Y-Y_hat|A=a]| <= epsilon for all groups (equalized residuals)
"""

from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np
import torch
from typing import List, Dict, Optional


class FairnessMetric(ABC):
    """
    Abstract base class for fairness metrics used in differentiable optimization layers.
    
    To create a custom fairness metric, subclass this and implement:
    1. create_constraints() - returns DPP-compliant cvxpy constraints
    2. compute_gap() - computes the actual gap value for monitoring
    3. Set requires_targets = True if your metric needs ground truth labels y
    4. Override create_selection_matrices() if you need custom selector structure
    
    The constraints must be affine and DPP-compliant:
    - yhat (cp.Variable) and y (cp.Parameter) are decision variable and parameter
    - selection_matrices are constant numpy arrays (0/1 values)
    - slack is a cp.Parameter for the tolerance
    - Use cp.multiply(yhat, selector) where selector is constant numpy array
    
    Attributes:
        num_protected_attrs: Number of protected attributes
        requires_targets: Whether this metric needs ground truth labels y (for selection matrices or constraints)
        requires_y_in_constraints: Whether y should be a cvxpy Parameter in the optimization problem
                                   (defaults to requires_targets, but can be overridden)
    """
    
    requires_targets: bool = False
    requires_y_in_constraints: bool = False  # Whether y is used in cvxpy constraints
    
    def __init__(self, num_protected_attrs: int = 1):
        """
        Initialize the fairness metric.
        
        Args:
            num_protected_attrs: Number of protected attributes to enforce fairness over
        """
        self.num_protected_attrs = num_protected_attrs
    
    def create_selection_matrices(
        self,
        x: 'torch.Tensor',
        y: Optional['torch.Tensor'],
        protected_attr_idx: List[int]
    ) -> List[np.ndarray]:
        """
        Create selection matrices for the fairness constraints.
        
        Override this method if your metric needs custom selector structure
        (e.g., EqualizedOdds needs selectors conditioned on outcomes).
        
        Default: Creates 2 selectors per attribute [A=0, A=1]
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            y: Target tensor of shape (batch_size,) or (batch_size, 1), may be None
            protected_attr_idx: List of column indices for protected attributes
            
        Returns:
            List of numpy arrays with 0/1 entries for group selection
        """
        selection_matrices = []
        
        for attr_idx in protected_attr_idx:
            protected_vals = x[:, attr_idx].cpu().numpy()
            
            selector_0 = (protected_vals == 0).astype(np.float32)
            selector_1 = (protected_vals == 1).astype(np.float32)
            
            selection_matrices.append(selector_0)
            selection_matrices.append(selector_1)
        
        return selection_matrices
    
    @abstractmethod
    def create_constraints(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        slack: cp.Parameter,
        y: Optional[cp.Parameter] = None
    ) -> List:
        """
        Create cvxpy constraints for the fairness metric.
        
        This method must return DPP-compliant constraints. The key rules are:
        - yhat is a cp.Variable (the predictions being optimized)
        - selection_matrices are constant numpy arrays with 0/1 entries
        - slack is a cp.Parameter (the fairness tolerance epsilon)
        - y is a cp.Parameter (targets, only provided if requires_targets=True)
        
        For DPP compliance:
        - Use cp.multiply(yhat, selector) where selector is a constant numpy array
        - Divide by constant scalars (e.g., n_0 = selector.sum())
        - Don't multiply parameters with each other
        
        Args:
            yhat: cvxpy Variable of shape (batch_size,) - predictions to constrain
            selection_matrices: List of numpy arrays with 0/1 entries for group selection
                               Structure: [attr0_group0, attr0_group1, attr1_group0, ...]
                               Each array has shape (batch_size,)
            slack: cvxpy Parameter (scalar) for the tolerance epsilon
            y: cvxpy Parameter of shape (batch_size,) for targets (if requires_targets=True)
            
        Returns:
            List of cvxpy constraints
        """
        pass
    
    @abstractmethod
    def compute_gap(
        self,
        predictions: np.ndarray,
        targets: Optional[np.ndarray],
        protected_indicators: Dict[int, np.ndarray]
    ) -> float:
        """
        Compute the fairness gap for monitoring during training.
        
        This should return a scalar representing how much the current predictions
        violate the fairness constraint. Used for logging, not for optimization.
        
        Args:
            predictions: numpy array of predictions, shape (n_samples,) or (n_samples, 1)
            targets: numpy array of targets (may be None if not needed)
            protected_indicators: Dict mapping attr_idx -> numpy array of 0/1 indicators
            
        Returns:
            Fairness gap value (typically max across all protected attributes)
        """
        pass
    
    def create_primal_dual_penalty(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        y: Optional[cp.Parameter] = None
    ) -> cp.Expression:
        """
        Create the penalty expression for primal-dual optimization.
        
        By default, this creates an auxiliary variable that upper-bounds all
        constraint violations. Override for custom penalty structures.
        
        Args:
            yhat: cvxpy Variable for predictions
            selection_matrices: List of selection matrices
            y: cvxpy Parameter for targets (if needed)
            
        Returns:
            cvxpy Expression representing the fairness penalty
        """
        # Default implementation: create max_gap variable
        # Subclasses can override for different penalty structures
        raise NotImplementedError(
            "Default primal-dual penalty not implemented. "
            "Override create_primal_dual_penalty() or use built-in metrics."
        )


class MeanPredictionParity(FairnessMetric):
    """
    Mean Prediction Parity (Demographic Parity for continuous predictions).
    
    Constraint: |E[Y_hat | A=0] - E[Y_hat | A=1]| <= epsilon for each protected attribute
    
    This ensures that the average prediction is similar across groups,
    regardless of the true labels
    """
    
    requires_targets = False
    
    def create_constraints(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        slack: cp.Parameter,
        y: Optional[cp.Parameter] = None
    ) -> List:
        constraints = []
        
        for attr_i in range(self.num_protected_attrs):
            selector_0 = selection_matrices[attr_i * 2]
            selector_1 = selection_matrices[attr_i * 2 + 1]
            
            n_0 = selector_0.sum()
            n_1 = selector_1.sum()
            
            if n_0 < 1 or n_1 < 1:
                continue
            
            # Mean predictions for each group
            # DPP-compliant: yhat (Variable) * selector (constant) / n (constant)
            mean_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
            mean_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
            
            # |mean_0 - mean_1| <= slack
            constraints += [
                mean_0 - mean_1 <= slack,
                mean_1 - mean_0 <= slack,
            ]
        
        return constraints
    
    def compute_gap(
        self,
        predictions: np.ndarray,
        targets: Optional[np.ndarray],
        protected_indicators: Dict[int, np.ndarray]
    ) -> float:
        predictions = predictions.flatten()
        max_gap = 0.0
        
        for attr_idx, indicators in protected_indicators.items():
            mask_0 = indicators == 0
            mask_1 = indicators == 1
            
            if mask_0.sum() == 0 or mask_1.sum() == 0:
                continue
            
            mean_0 = predictions[mask_0].mean()
            mean_1 = predictions[mask_1].mean()
            gap = abs(mean_0 - mean_1)
            max_gap = max(max_gap, gap)
        
        return max_gap
    
    def create_primal_dual_penalty(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        y: Optional[cp.Parameter] = None
    ) -> tuple:
        """
        Create penalty for primal-dual formulation.
        
        Returns:
            Tuple of (max_gap variable, constraints that define it)
        """
        max_gap = cp.Variable(1, nonneg=True)
        constraints = []
        
        for attr_i in range(self.num_protected_attrs):
            selector_0 = selection_matrices[attr_i * 2]
            selector_1 = selection_matrices[attr_i * 2 + 1]
            
            n_0 = selector_0.sum()
            n_1 = selector_1.sum()
            
            if n_0 < 1 or n_1 < 1:
                continue
            
            mean_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
            mean_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
            
            # max_gap >= |mean_0 - mean_1|
            constraints += [
                max_gap >= mean_0 - mean_1,
                max_gap >= mean_1 - mean_0,
            ]
        
        return max_gap, constraints


class MeanResidualFairness(FairnessMetric):
    """
    Mean Residual Fairness (Equalized Residuals).
    
    Constraint: |E[Y - Y_hat | A=a]| <= epsilon for each group a
    
    This ensures the model doesn't systematically over-predict or under-predict
    for any protected group
            
    Example:
        metric = MeanResidualFairness(num_protected_attrs=1)
        model = FairModel(..., fairness_metric=metric)
    """
    
    requires_targets = True
    requires_y_in_constraints = True  # y is used in cvxpy constraints for residual computation
    
    def create_constraints(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        slack: cp.Parameter,
        y: Optional[cp.Parameter] = None
    ) -> List:
        if y is None:
            raise ValueError("MeanResidualFairness requires targets y")
        
        constraints = []
        
        for attr_i in range(self.num_protected_attrs):
            selector_0 = selection_matrices[attr_i * 2]
            selector_1 = selection_matrices[attr_i * 2 + 1]
            
            n_0 = selector_0.sum()
            n_1 = selector_1.sum()
            
            if n_0 < 1 or n_1 < 1:
                continue
            
            # Mean predictions and targets for each group
            # DPP-compliant: Variable * constant, Parameter * constant
            mean_pred_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
            mean_y_0 = cp.sum(cp.multiply(y, selector_0)) / n_0
            
            mean_pred_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
            mean_y_1 = cp.sum(cp.multiply(y, selector_1)) / n_1
            
            # |mean_pred - mean_y| <= slack for each group
            # Equivalent to: |mean_residual| <= slack
            constraints += [
                mean_pred_0 - mean_y_0 <= slack,
                mean_pred_0 - mean_y_0 >= -slack,
                mean_pred_1 - mean_y_1 <= slack,
                mean_pred_1 - mean_y_1 >= -slack,
            ]
        
        return constraints
    
    def compute_gap(
        self,
        predictions: np.ndarray,
        targets: Optional[np.ndarray],
        protected_indicators: Dict[int, np.ndarray]
    ) -> float:
        if targets is None:
            raise ValueError("MeanResidualFairness.compute_gap requires targets")
        
        predictions = predictions.flatten()
        targets = targets.flatten()
        max_gap = 0.0
        
        for attr_idx, indicators in protected_indicators.items():
            mask_0 = indicators == 0
            mask_1 = indicators == 1
            
            if mask_0.sum() == 0 or mask_1.sum() == 0:
                continue
            
            residual_0 = (targets[mask_0] - predictions[mask_0]).mean()
            residual_1 = (targets[mask_1] - predictions[mask_1]).mean()
            gap = max(abs(residual_0), abs(residual_1))
            max_gap = max(max_gap, gap)
        
        return max_gap
    
    def create_primal_dual_penalty(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        y: Optional[cp.Parameter] = None
    ) -> tuple:
        if y is None:
            raise ValueError("MeanResidualFairness requires targets y for primal-dual")
        
        max_gap = cp.Variable(1, nonneg=True)
        constraints = []
        
        for attr_i in range(self.num_protected_attrs):
            selector_0 = selection_matrices[attr_i * 2]
            selector_1 = selection_matrices[attr_i * 2 + 1]
            
            n_0 = selector_0.sum()
            n_1 = selector_1.sum()
            
            if n_0 < 1 or n_1 < 1:
                continue
            
            mean_pred_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
            mean_y_0 = cp.sum(cp.multiply(y, selector_0)) / n_0
            
            mean_pred_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
            mean_y_1 = cp.sum(cp.multiply(y, selector_1)) / n_1
            
            residual_0 = mean_pred_0 - mean_y_0
            residual_1 = mean_pred_1 - mean_y_1
            
            # max_gap >= |residual| for each group
            constraints += [
                max_gap >= residual_0,
                max_gap >= -residual_0,
                max_gap >= residual_1,
                max_gap >= -residual_1,
            ]
        
        return max_gap, constraints


class EqualizedOdds(FairnessMetric):
    """
    Equalized Odds (based on expected values).
    
    Constraint: |E[Y_hat | Y=y, A=0] - E[Y_hat | Y=y, A=1]| <= epsilon for each outcome y
    
    This ensures that the average prediction is similar across protected groups,
    separately for each true outcome class
    
    Requires binary ground truth labels,
    
    Note: This metric creates constraints for each (outcome, group) combination,
    which requires computing selection matrices that depend on both protected
    attribute and outcome. The selection_matrices should be structured as:
    [attr0_y0_group0, attr0_y0_group1, attr0_y1_group0, attr0_y1_group1, ...]
    
    Example:
        metric = EqualizedOdds(num_protected_attrs=1)
        model = FairModel(..., fairness_metric=metric)
    """
    
    requires_targets = True
    
    def create_selection_matrices(
        self,
        x: 'torch.Tensor',
        y: Optional['torch.Tensor'],
        protected_attr_idx: List[int]
    ) -> List[np.ndarray]:
        """
        Create selection matrices for EqualizedOdds.
        
        Creates 4 selectors per attribute: [y=0&A=0, y=0&A=1, y=1&A=0, y=1&A=1]
        """
        if y is None:
            raise ValueError("EqualizedOdds requires targets y for selection matrices")
        
        # Ensure y is 1D
        y_np = y.squeeze().cpu().numpy()
        y_binary = (y_np > 0.5).astype(np.float32)
        
        selection_matrices = []
        
        for attr_idx in protected_attr_idx:
            protected_vals = x[:, attr_idx].cpu().numpy()
            
            # Create 4 selectors: (Y=0, A=0), (Y=0, A=1), (Y=1, A=0), (Y=1, A=1)
            sel_y0_g0 = ((y_binary == 0) & (protected_vals == 0)).astype(np.float32)
            sel_y0_g1 = ((y_binary == 0) & (protected_vals == 1)).astype(np.float32)
            sel_y1_g0 = ((y_binary == 1) & (protected_vals == 0)).astype(np.float32)
            sel_y1_g1 = ((y_binary == 1) & (protected_vals == 1)).astype(np.float32)
            
            selection_matrices.extend([sel_y0_g0, sel_y0_g1, sel_y1_g0, sel_y1_g1])
        
        return selection_matrices
    
    def create_constraints(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        slack: cp.Parameter,
        y: Optional[cp.Parameter] = None
    ) -> List:
        """
        Note: For EqualizedOdds, selection_matrices should be pre-computed
        to include outcome conditioning. Each attribute contributes 4 selectors:
        [y=0 & A=0, y=0 & A=1, y=1 & A=0, y=1 & A=1]
        
        This is handled by create_selection_matrices().
        The y parameter is not used here since outcome info is in the selectors.
        """
        constraints = []
        
        # For EqualizedOdds, we expect 4 selectors per attribute
        # (2 outcomes × 2 groups)
        selectors_per_attr = 4
        
        for attr_i in range(self.num_protected_attrs):
            base_idx = attr_i * selectors_per_attr
            
            # For Y=0: compare groups 0 and 1
            sel_y0_g0 = selection_matrices[base_idx]
            sel_y0_g1 = selection_matrices[base_idx + 1]
            
            # For Y=1: compare groups 0 and 1  
            sel_y1_g0 = selection_matrices[base_idx + 2]
            sel_y1_g1 = selection_matrices[base_idx + 3]
            
            # Y=0 constraint
            n_y0_g0 = sel_y0_g0.sum()
            n_y0_g1 = sel_y0_g1.sum()
            
            if n_y0_g0 >= 1 and n_y0_g1 >= 1:
                mean_y0_g0 = cp.sum(cp.multiply(yhat, sel_y0_g0)) / n_y0_g0
                mean_y0_g1 = cp.sum(cp.multiply(yhat, sel_y0_g1)) / n_y0_g1
                constraints += [
                    mean_y0_g0 - mean_y0_g1 <= slack,
                    mean_y0_g1 - mean_y0_g0 <= slack,
                ]
            
            # Y=1 constraint
            n_y1_g0 = sel_y1_g0.sum()
            n_y1_g1 = sel_y1_g1.sum()
            
            if n_y1_g0 >= 1 and n_y1_g1 >= 1:
                mean_y1_g0 = cp.sum(cp.multiply(yhat, sel_y1_g0)) / n_y1_g0
                mean_y1_g1 = cp.sum(cp.multiply(yhat, sel_y1_g1)) / n_y1_g1
                constraints += [
                    mean_y1_g0 - mean_y1_g1 <= slack,
                    mean_y1_g1 - mean_y1_g0 <= slack,
                ]
        
        return constraints
    
    def compute_gap(
        self,
        predictions: np.ndarray,
        targets: Optional[np.ndarray],
        protected_indicators: Dict[int, np.ndarray]
    ) -> float:
        if targets is None:
            raise ValueError("EqualizedOdds.compute_gap requires targets")
        
        predictions = predictions.flatten()
        targets = targets.flatten()
        # Binarize targets
        targets_binary = (targets > 0.5).astype(int)
        
        max_gap = 0.0
        
        for attr_idx, indicators in protected_indicators.items():
            for y_val in [0, 1]:
                y_mask = targets_binary == y_val
                
                mask_0 = (indicators == 0) & y_mask
                mask_1 = (indicators == 1) & y_mask
                
                if mask_0.sum() == 0 or mask_1.sum() == 0:
                    continue
                
                mean_0 = predictions[mask_0].mean()
                mean_1 = predictions[mask_1].mean()
                gap = abs(mean_0 - mean_1)
                max_gap = max(max_gap, gap)
                
        return max_gap
    
    def create_primal_dual_penalty(
        self,
        yhat: cp.Variable,
        selection_matrices: List[np.ndarray],
        y: Optional[cp.Parameter] = None
    ) -> tuple:
        max_gap = cp.Variable(1, nonneg=True)
        constraints = []
        
        selectors_per_attr = 4
        
        for attr_i in range(self.num_protected_attrs):
            base_idx = attr_i * selectors_per_attr
            
            sel_y0_g0 = selection_matrices[base_idx]
            sel_y0_g1 = selection_matrices[base_idx + 1]
            sel_y1_g0 = selection_matrices[base_idx + 2]
            sel_y1_g1 = selection_matrices[base_idx + 3]
            
            # Y=0 gap
            n_y0_g0, n_y0_g1 = sel_y0_g0.sum(), sel_y0_g1.sum()
            if n_y0_g0 >= 1 and n_y0_g1 >= 1:
                mean_y0_g0 = cp.sum(cp.multiply(yhat, sel_y0_g0)) / n_y0_g0
                mean_y0_g1 = cp.sum(cp.multiply(yhat, sel_y0_g1)) / n_y0_g1
                constraints += [
                    max_gap >= mean_y0_g0 - mean_y0_g1,
                    max_gap >= mean_y0_g1 - mean_y0_g0,
                ]
            
            # Y=1 gap
            n_y1_g0, n_y1_g1 = sel_y1_g0.sum(), sel_y1_g1.sum()
            if n_y1_g0 >= 1 and n_y1_g1 >= 1:
                mean_y1_g0 = cp.sum(cp.multiply(yhat, sel_y1_g0)) / n_y1_g0
                mean_y1_g1 = cp.sum(cp.multiply(yhat, sel_y1_g1)) / n_y1_g1
                constraints += [
                    max_gap >= mean_y1_g0 - mean_y1_g1,
                    max_gap >= mean_y1_g1 - mean_y1_g0,
                ]
        
        return max_gap, constraints
