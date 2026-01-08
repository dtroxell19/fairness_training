"""
Fair Neural Network with Differentiable Fairness Constraints

This module implements a neural network with an embedded cvxpylayers optimization 
layer that enforces group fairness constraints during training through differentiable
convex optimization.

Key Features:
- Differentiable fairness layer using cvxpylayers
- Training: Always uses hard per-batch fairness constraints (requires batch_size >= b_tau)
- Inference: Two regimes based on batch size vs threshold b_tau:
  * Large batches (>= b_tau): Hard per-batch fairness constraints
  * Small batches (< b_tau): Online primal-dual algorithm
- Support for multiple protected attributes (marginal fairness - constraints per attribute)
- Extensible: Support for custom fairness metrics via FairnessMetric interface
"""

import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional, Union, Tuple, List

from fairness_metrics import FairnessMetric
from utils import get_fairness_metric

class FairModel(nn.Module):
    """
    Neural network with differentiable fairness constraints.
    
    This model projects predictions from a standard neural network through a 
    cvxpylayers optimization layer that enforces marginal fairness constraints
    while minimizing distortion from the original predictions.
    
    Marginal Fairness: For each protected attribute independently, the model
    ensures constraints across groups
    
    Training: Always uses hard per-batch constraints (batch_size should be >= b_tau)
    Inference: Uses hard constraints if batch_size >= b_tau, otherwise uses online
               primal-dual algorithm
    
    Args:
        input_dim (int): Number of input features
        hidden_dims (List[int]): List of hidden layer dimensions (optional if custom_network provided)
        output_dim (int): Output dimension (typically 1 for binary classification)
        protected_attr_idx (Union[int, List[int]]): Index or list of indices of protected attribute columns
        prediction_bounds (Tuple[float, float]): (lower, upper) bounds for predictions
        fairness_tolerance (float): Target fairness tolerance epsilon
        b_tau (int): Batch size threshold for inference - above uses hard constraints, below uses primal-dual
        eta_0 (float): Initial dual step size for inference primal-dual updates
        activation (str): Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        fairness_metric: Either a string ('mean_pred', 'mean_residual') or a FairnessMetric instance
        custom_network (nn.Module): Optional custom network architecture to use instead of default
        
    Example:
        # Basic usage with string metric
        model = FairModel(input_dim=20, hidden_dims=[64, 32], protected_attr_idx=0)
        
        # Multiple protected attributes (marginal fairness)
        model = FairModel(input_dim=20, hidden_dims=[64, 32], protected_attr_idx=[0, 1])
        
        # With custom network
        custom_net = MyCustomNetwork(input_dim=20, output_dim=1)
        model = FairModel(input_dim=20, protected_attr_idx=0, custom_network=custom_net)
        
        # With custom fairness metric
        from fairness_metrics import FairnessMetric
        class MyMetric(FairnessMetric):
            ...
        model = FairModel(input_dim=20, hidden_dims=[64, 32], 
                         protected_attr_idx=0, fairness_metric=MyMetric(num_protected_attrs=1))
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        protected_attr_idx: Union[int, List[int]] = 0,
        prediction_bounds: Tuple[float, float] = (0.0, 1.0),
        fairness_tolerance: float = 0.05,
        b_tau: int = 2000,
        eta_0: float = 0.5,
        activation: str = 'relu',
        fairness_metric: Union[str, FairnessMetric] = 'mean_pred',
        custom_network: Optional[nn.Module] = None,
        #will be converted to fairness_metric
        fairness_criterion: Optional[str] = None
    ):
        super(FairModel, self).__init__()
        
        # Model configuration
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Handle 1 or 2 protected attributes
        if isinstance(protected_attr_idx, int):
            self.protected_attr_idx = [protected_attr_idx]
        else:
            self.protected_attr_idx = list(protected_attr_idx)
        
        self.num_protected_attrs = len(self.protected_attr_idx)
        if self.num_protected_attrs > 2:
            raise ValueError("FairModel currently only supports up to 2 protected attributes.")
        
        self.lb, self.ub = prediction_bounds
        
        # Handle legacy fairness_criterion parameter
        if fairness_criterion is not None:
            import warnings
            warnings.warn(
                "fairness_criterion is deprecated, use fairness_metric instead",
                DeprecationWarning
            )
            fairness_metric = fairness_criterion
        
        self.fairness_metric = get_fairness_metric(fairness_metric, self.num_protected_attrs)
        
        self.requires_targets = self.fairness_metric.requires_targets
        self.requires_y_in_constraints = self.fairness_metric.requires_y_in_constraints
        self.fairness_tolerance = fairness_tolerance  # epsilon in paper
        self.b_tau = b_tau
        
        # Primal-dual variables for inferendce (when batch_size < b_tau)
        self.lambda_dual = 0.0
        self.eta_0 = eta_0
        self.dual_update_count = 0
        
        # Tracking for convergence analysis (inference)
        self.cumulative_samples = 0
        self.cumulative_weighted_violation = 0.0
        self.lambda_max = 0.0
        
        # Build or use custom network
        if custom_network is not None:
            self.ffnn = custom_network
            print(f"Using custom network architecture")
        else:
            if hidden_dims is None:
                raise ValueError("Either provide custom_network or specify hidden_dims")
            self.ffnn = self._build_network(activation)
            self._initialize_weights()
        
        # Placeholder for cvxpy layer (created dynamically later)
        self.cvxpylayer = None
        
    def _build_network(self, activation: str) -> nn.Sequential:
        """Construct the feedforward neural network."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        act_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }.get(activation.lower(), nn.ReLU())
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_fn)
                
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            
    def forward(
        self, 
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        inference: bool = False
    ) -> torch.Tensor:
        """
        Forward pass that routes to inference or training implementations.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            y: Target tensor (required if fairness_metric.requires_targets is True)
            inference: If True, use inference mode (may use primal-dual for small batches)
            
        Returns:
            Fair predictions of shape (batch_size, output_dim)
        """
        # Get raw predictions from neural network
        y_hat = self.ffnn(x)
        
        # Check if targets are needed
        if self.requires_targets and y is None:
            raise ValueError(
                f"Target y must be provided when using {type(self.fairness_metric).__name__} "
                f"(requires_targets=True)"
            )
        
        batch_size = len(x)
        
        # TRAINING: Always use hard constraints (batch_size should be >= b_tau)
        if not inference:
            return self._forward_hard_constraint(x, y_hat, y)
        
        # INFERENCE: Choose algorithm based on batch size
        if batch_size >= self.b_tau:
            return self._forward_hard_constraint(x, y_hat, y)
        else:
            return self._forward_primal_dual(x, y_hat, y)
    
    def _forward_hard_constraint(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply hard fairness constraints using cvxpylayers."""
        batch_size = len(x)
        
        # Create selection matrices using the metric's method (allows metric-specific logic)
        selection_matrices = self.fairness_metric.create_selection_matrices(
            x, y, self.protected_attr_idx
        )
        
        self.cvxpylayer = self._create_hard_constraint_layer(batch_size, selection_matrices)
        
        # Build parameter list
        params = [y_hat.squeeze(1)]
        if self.requires_y_in_constraints:
            params.append(y.squeeze(1) if y.dim() > 1 else y)
        params.append(torch.tensor([self.fairness_tolerance], dtype=torch.float32))
        
        try:
            ytilde = self.cvxpylayer(*params)[0]
        except Exception as e:
            self._print_solver_debug_info(e, x, y_hat, y, selection_matrices, batch_size)
            raise
        
        return ytilde.unsqueeze(1)
    
    def _forward_primal_dual(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Small-batch inference using online primal-dual algorithm.
        
        Individual batches may violate fairness constraints, but aggregate
        fairness is guaranteed over time
        """
        batch_size = len(x)
        
        # Create selection matrices using the metric's method
        selection_matrices = self.fairness_metric.create_selection_matrices(
            x, y, self.protected_attr_idx
        )
        
        self.cvxpylayer = self._create_primal_dual_layer(
            batch_size, selection_matrices, self.lambda_dual
        )
        
        # Build parameter list
        params = [y_hat.squeeze(1)]
        if self.requires_y_in_constraints:
            params.append(y.squeeze(1) if y.dim() > 1 else y)
        
        # Solve primal problem
        try:
            ytilde = self.cvxpylayer(*params)[0]
        except Exception as e:
            self._print_solver_debug_info(e, x, y_hat, y, selection_matrices, batch_size)
            raise
        
        ytilde_unsqueezed = ytilde.unsqueeze(1)
        
        # Compute fairness gap for this batch
        with torch.no_grad():
            predictions_np = ytilde.detach().cpu().numpy()
            targets_np = y.detach().cpu().numpy() if y is not None else None
            protected_indicators = {
                attr_idx: x[:, attr_idx].cpu().numpy() 
                for attr_idx in self.protected_attr_idx
            }
            
            gap = self.fairness_metric.compute_gap(predictions_np, targets_np, protected_indicators)
            violation = gap - self.fairness_tolerance
            weighted_violation = batch_size * violation
            
            # Dual update 
            self.dual_update_count += 1
            eta_t = self.eta_0 / np.sqrt(self.dual_update_count)
            self.lambda_dual = max(0.0, self.lambda_dual + eta_t * weighted_violation)
            
            # Track statistics
            self.cumulative_samples += batch_size
            self.cumulative_weighted_violation += weighted_violation
            self.lambda_max = max(self.lambda_max, self.lambda_dual)
        
        return ytilde_unsqueezed
    
    def _create_hard_constraint_layer(
        self,
        batch_size: int,
        selection_matrices: List[np.ndarray]
    ) -> CvxpyLayer:
        """
        Create cvxpylayers layer with hard fairness constraints.
        
        Delegates constraint creation to the fairness metric.
        """
        yhat = cp.Variable(batch_size)
        raw = cp.Parameter(batch_size)
        slack = cp.Parameter(1, nonneg=True)
        
        constraints = [yhat >= self.lb, yhat <= self.ub]
        
        # Add target parameter if needed for constraints
        if self.requires_y_in_constraints:
            y = cp.Parameter(batch_size)
            fairness_constraints = self.fairness_metric.create_constraints(
                yhat, selection_matrices, slack, y
            )
        else:
            y = None
            fairness_constraints = self.fairness_metric.create_constraints(
                yhat, selection_matrices, slack
            )
        
        constraints.extend(fairness_constraints)
        
        # Objective: minimize discrepancy function
        objective = cp.Minimize(cp.sum_squares(yhat - raw))
        problem = cp.Problem(objective, constraints)
        
        # Build parameter list
        if self.requires_y_in_constraints:
            return CvxpyLayer(problem, parameters=[raw, y, slack], variables=[yhat])
        else:
            return CvxpyLayer(problem, parameters=[raw, slack], variables=[yhat])
    
    def _create_primal_dual_layer(
        self,
        batch_size: int,
        selection_matrices: List[np.ndarray],
        lambda_val: float
    ) -> CvxpyLayer:
        """
        Create cvxpylayers layer for primal-dual inference.
        
        Moves fairness constraints to objective as penalty.
        """
        yhat = cp.Variable(batch_size)
        max_gap = cp.Variable(1, nonneg=True)
        raw = cp.Parameter(batch_size)
        
        # Box constraints
        constraints = [yhat >= self.lb, yhat <= self.ub]
        
        # Get penalty constraints from metric
        if self.requires_y_in_constraints:
            y = cp.Parameter(batch_size)
            max_gap_var, penalty_constraints = self.fairness_metric.create_primal_dual_penalty(
                yhat, selection_matrices, y
            )
        else:
            y = None
            max_gap_var, penalty_constraints = self.fairness_metric.create_primal_dual_penalty(
                yhat, selection_matrices
            )
        
        # Use max_gap from metric if returned, otherwise use our own
        if max_gap_var is not None:
            max_gap = max_gap_var
            constraints.extend(penalty_constraints)
        
        # Objective: distortion + penalty
        objective = cp.Minimize(
            cp.sum_squares(yhat - raw) + lambda_val * batch_size * max_gap
        )
        problem = cp.Problem(objective, constraints)
        
        if self.requires_y_in_constraints:
            return CvxpyLayer(problem, parameters=[raw, y], variables=[yhat])
        else:
            return CvxpyLayer(problem, parameters=[raw], variables=[yhat])
    
    def _print_solver_debug_info(
        self,
        error: Exception,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: Optional[torch.Tensor],
        selection_matrices: List[np.ndarray],
        batch_size: int
    ):
        """Print debug information when solver fails"""
        print("\n" + "="*80)
        print("CVXPY SOLVER FAILED")
        print("="*80)
        print(f"Error: {error}")
        print(f"\nBatch size: {batch_size}")
        print(f"Fairness tolerance: {self.fairness_tolerance}")
        print(f"Metric: {type(self.fairness_metric).__name__}")
        print(f"\nRaw predictions:")
        print(f"  Shape: {y_hat.squeeze(1).shape}")
        print(f"  Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}")
        print(f"  Mean: {y_hat.mean().item():.4f}, Std: {y_hat.std().item():.4f}")
        
        print(f"\nGroup distributions:")
        for attr_i, attr_idx in enumerate(self.protected_attr_idx):
            selector_0 = selection_matrices[attr_i * 2]
            selector_1 = selection_matrices[attr_i * 2 + 1]
            n_0 = selector_0.sum()
            n_1 = selector_1.sum()
            print(f"  Attribute {attr_idx}: Group 0 = {n_0:.0f}, Group 1 = {n_1:.0f}")
            
            if n_0 > 0 and n_1 > 0:
                mask_0 = torch.from_numpy(selector_0.astype(bool))
                mask_1 = torch.from_numpy(selector_1.astype(bool))
                mean_0 = y_hat.squeeze(1)[mask_0].mean().item()
                mean_1 = y_hat.squeeze(1)[mask_1].mean().item()
                gap = abs(mean_0 - mean_1)
                print(f"    Unconstrained gap: {gap:.6f} (tolerance: {self.fairness_tolerance})")
        print("="*80 + "\n")
    
    def reset_inference_state(self):
        """Reset primal-dual state for inference. Call before new inference sequence"""
        self.lambda_dual = 0.0
        self.dual_update_count = 0
        self.cumulative_samples = 0
        self.cumulative_weighted_violation = 0.0
        self.lambda_max = 0.0
    
    def get_aggregate_fairness_stats(
        self,
        data_loader,
        reset_before: bool = True
    ) -> dict:
        """
        Compute aggregate fairness statistics over a data loader
        
        Uses inference mode with primal-dual algorithm for small batches.
        
        Args:
            data_loader: PyTorch DataLoader
            reset_before: If True, reset inference state before evaluation
            
        Returns:
            Dictionary with aggregate statistics
        """
        if reset_before:
            self.reset_inference_state()
        
        self.eval()
        
        all_predictions = []
        all_targets = []
        all_protected = {attr_idx: [] for attr_idx in self.protected_attr_idx}
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                # Check if both groups present for all attributes
                skip_batch = False
                for attr_idx in self.protected_attr_idx:
                    if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                       (batch_x[:, attr_idx] == 1).sum() < 1:
                        skip_batch = True
                        break
                
                if skip_batch:
                    continue
                
                # Forward pass in inference mode
                if self.requires_targets:
                    predictions = self(batch_x, y=batch_y, inference=True)
                else:
                    predictions = self(batch_x, inference=True)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
                
                for attr_idx in self.protected_attr_idx:
                    all_protected[attr_idx].append(batch_x[:, attr_idx].cpu())
        
        # Aggregate
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        protected_indicators = {}
        for attr_idx in self.protected_attr_idx:
            protected_indicators[attr_idx] = torch.cat(all_protected[attr_idx], dim=0).numpy()
        
        aggregate_gap = self.fairness_metric.compute_gap(
            all_predictions, all_targets, protected_indicators
        )
        
        # per-attribute gaps
        per_attr_gaps = {}
        for attr_idx in self.protected_attr_idx:
            single_attr_indicators = {attr_idx: protected_indicators[attr_idx]}
            per_attr_gaps[attr_idx] = self.fairness_metric.compute_gap(
                all_predictions, all_targets, single_attr_indicators
            )
        
        return {
            'aggregate_gap': aggregate_gap,
            'per_attribute_gaps': per_attr_gaps,
            'lambda_max': self.lambda_max,
            'lambda_final': self.lambda_dual,
            'total_samples': self.cumulative_samples,
            'num_batches': self.dual_update_count,
            'fairness_tolerance': self.fairness_tolerance
        }
