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
  * Small batches (< b_tau): Online primal-dual algorithm (Algorithm 1 from paper)
- Provable aggregate fairness guarantees via Theorem 2.2 for small-batch inference
- Support for multiple protected attributes (MARGINAL fairness - constraints per attribute)
- Support for both mean prediction and mean residual fairness criteria
"""

import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional, Union, Tuple, List


class FairModel(nn.Module):
    """
    Neural network with differentiable fairness constraints.
    
    This model projects predictions from a standard neural network through a 
    cvxpylayers optimization layer that enforces marginal fairness constraints
    while minimizing distortion from the original predictions.
    
    Marginal Fairness: For EACH protected attribute independently, the model
    ensures equal treatment across groups. This is more flexible than intersectional
    fairness and scales linearly with the number of protected attributes.
    
    Training: Always uses hard per-batch constraints (batch_size should be >= b_tau)
    Inference: Uses hard constraints if batch_size >= b_tau, otherwise uses
               primal-dual algorithm for provable aggregate fairness (Theorem 2.2)
    
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
        fairness_criterion (str): 'mean_pred' for equal mean predictions, 
                                   'mean_residual' for equal mean residuals
        custom_network (nn.Module): Optional custom network architecture to use instead of default
        
    Example:
        # Basic usage
        model = FairModel(input_dim=20, hidden_dims=[64, 32], protected_attr_idx=0)
        
        # Multiple protected attributes (marginal fairness)
        model = FairModel(input_dim=20, hidden_dims=[64, 32], protected_attr_idx=[0, 1, 3])
        
        # With custom network
        custom_net = MyCustomNetwork(input_dim=20, output_dim=1)
        model = FairModel(input_dim=20, protected_attr_idx=0, custom_network=custom_net)
        
        # Mean residual fairness (for regression)
        model = FairModel(input_dim=20, hidden_dims=[64, 32], 
                         protected_attr_idx=0, fairness_criterion='mean_residual')
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
        fairness_criterion: str = 'mean_pred',
        custom_network: Optional[nn.Module] = None
    ):
        super(FairModel, self).__init__()
        
        # Model configuration
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Handle single or multiple protected attributes
        if isinstance(protected_attr_idx, int):
            self.protected_attr_idx = [protected_attr_idx]
        else:
            self.protected_attr_idx = list(protected_attr_idx)
        
        self.num_protected_attrs = len(self.protected_attr_idx)
        self.lb, self.ub = prediction_bounds
        
        # Fairness criterion
        if fairness_criterion not in ['mean_pred', 'mean_residual']:
            raise ValueError(f"fairness_criterion must be 'mean_pred' or 'mean_residual', got {fairness_criterion}")
        self.fairness_criterion = fairness_criterion
        
        # Fairness constraint parameters
        self.fairness_tolerance = fairness_tolerance  # epsilon in paper
        self.b_tau = b_tau  # Threshold for switching between regimes (inference only)
        
        # Primal-dual variables for INFERENCE (when batch_size < b_tau)
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
            print(f"✓ Using custom network architecture")
        else:
            if hidden_dims is None:
                raise ValueError("Either provide custom_network or specify hidden_dims")
            self.ffnn = self._build_network(activation)
            self._initialize_weights()
        
        # Placeholder for cvxpy layer (created dynamically in forward pass)
        self.cvxpylayer = None
        
    def _build_network(self, activation: str) -> nn.Sequential:
        """Construct the feedforward neural network."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        # Activation function mapping
        act_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }.get(activation.lower(), nn.ReLU())
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(act_fn)
                
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def get_adaptive_eta(self) -> float:
        """Compute adaptive step size using 1/sqrt(t) schedule for inference."""
        return self.eta_0 / np.sqrt(max(1, self.dual_update_count))
            
    def forward(
        self, 
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        inference: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with fairness projection.
        
        Training (inference=False): Always uses hard per-batch constraints
        Inference (inference=True): Uses hard constraints if batch >= b_tau,
                                    otherwise uses primal-dual algorithm
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            y: Target tensor (required if fairness_criterion='mean_residual')
            inference: If True, use inference mode (may use primal-dual for small batches)
            
        Returns:
            Fair predictions of shape (batch_size, output_dim)
        """
        # Get raw predictions from neural network
        y_hat = self.ffnn(x)
        
        # Check if targets are needed
        if self.fairness_criterion == 'mean_residual' and y is None:
            raise ValueError("Target y must be provided when fairness_criterion='mean_residual'")
        
        batch_size = len(x)
        
        if inference:
            # Inference mode: choose algorithm based on batch size
            if batch_size >= self.b_tau:
                # Large batch: use hard constraints
                return self._forward_hard_constraints(x, y_hat, y)
            else:
                # Small batch: use primal-dual algorithm
                return self._forward_primal_dual(x, y_hat, y)
        else:
            # Training mode: always use hard constraints
            return self._forward_hard_constraints(x, y_hat, y)
    
    def _forward_hard_constraints(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with hard per-batch MARGINAL fairness constraints.
        Used for all training and large-batch inference.
        
        Creates a single optimization problem with constraints for ALL protected attributes.
        """
        batch_size = len(x)
        
        # Create selection matrices for each protected attribute
        selection_matrices = self._create_selection_matrices(x)
        
        # Create and solve fairness projection with marginal constraints
        self.cvxpylayer = self._create_hard_constraint_layer_marginal(
            batch_size, selection_matrices
        )
        
        try:
            raw = y_hat.squeeze(-1)
            
            if self.fairness_criterion == 'mean_residual':
                y_flat = y.squeeze(-1) if y.dim() > 1 else y
                result = self.cvxpylayer(
                    raw,
                    y_flat,
                    torch.tensor([self.fairness_tolerance], dtype=torch.float32)
                )
            else:
                result = self.cvxpylayer(
                    raw,
                    torch.tensor([self.fairness_tolerance], dtype=torch.float32)
                )
            
            ytilde = result[0].unsqueeze(-1)
            
        except Exception as e:
            self._print_debug_info("HARD CONSTRAINTS", x, y_hat, selection_matrices, e)
            raise
        
        return ytilde
    
    def _forward_primal_dual(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass using primal-dual algorithm for small-batch inference.
        Implements Algorithm 1 from the paper with MARGINAL fairness.
        """
        batch_size = len(x)
        
        # Create selection matrices for all protected attributes
        selection_matrices = self._create_selection_matrices(x)
        
        # Get current lambda
        lambda_val = self.lambda_dual
        
        # Create and solve primal-dual projection with marginal constraints
        self.cvxpylayer = self._create_primal_dual_layer_marginal(
            batch_size, selection_matrices, lambda_val
        )
        
        try:
            raw = y_hat.squeeze(-1)
            
            if self.fairness_criterion == 'mean_residual':
                y_flat = y.squeeze(-1) if y.dim() > 1 else y
                result = self.cvxpylayer(raw, y_flat)
            else:
                result = self.cvxpylayer(raw)
            
            ytilde = result[0].unsqueeze(-1)
            
        except Exception as e:
            self._print_debug_info("PRIMAL-DUAL", x, y_hat, selection_matrices, e, lambda_val)
            raise
        
        # Compute fairness gap for dual update (max across all attributes)
        gap = self._compute_batch_fairness_gap(ytilde, x, y)
        violation = gap - self.fairness_tolerance
        weighted_violation = batch_size * violation
        
        # Dual update: λ = max(0, λ + η_t * weighted_violation)
        self.dual_update_count += 1
        eta_t = self.get_adaptive_eta()
        self.lambda_dual = max(0.0, self.lambda_dual + eta_t * weighted_violation)
        
        # Track statistics for Theorem 2.2 bound
        self.lambda_max = max(self.lambda_max, self.lambda_dual)
        self.cumulative_samples += batch_size
        self.cumulative_weighted_violation += weighted_violation
        
        return ytilde
    
    def _create_selection_matrices(self, x: torch.Tensor) -> List[np.ndarray]:
        """
        Create binary selection matrices for each protected attribute and group.
        
        Returns list of 2*num_protected_attrs numpy arrays of shape (batch_size,)
        with 0/1 float entries indicating group membership.
        
        For each attribute i: [selector_group0, selector_group1]
        Structure: [attr0_group0, attr0_group1, attr1_group0, attr1_group1, ...]
        """
        selection_matrices = []
        
        for attr_idx in self.protected_attr_idx:
            protected_vals = x[:, attr_idx].cpu().numpy()
            
            # Group 0 selector (1.0 where protected == 0, else 0.0)
            selector_0 = (protected_vals == 0).astype(np.float32)
            # Group 1 selector (1.0 where protected == 1, else 0.0)
            selector_1 = (protected_vals == 1).astype(np.float32)
            
            selection_matrices.append(selector_0)
            selection_matrices.append(selector_1)
        
        return selection_matrices
    
    def _create_hard_constraint_layer_marginal(
        self,
        batch_size: int,
        selection_matrices: List[np.ndarray]
    ) -> CvxpyLayer:
        """
        Create cvxpylayers optimization layer with MARGINAL fairness constraints.
        
        Uses a SINGLE decision variable yhat for all predictions, with selection
        matrices to compute group-wise statistics for each protected attribute.
        
        Solves:
            minimize    ||yhat - raw||^2
            subject to  lb <= yhat <= ub
                        For each protected attribute j:
                            |mean(yhat | A_j=0) - mean(yhat | A_j=1)| <= slack (mean_pred)
                            OR
                            |mean(y - yhat | A_j=0)| <= slack AND
                            |mean(y - yhat | A_j=1)| <= slack (mean_residual)
        """
        # Single decision variable for all predictions
        yhat = cp.Variable(batch_size)
        
        # Parameters
        raw = cp.Parameter(batch_size)
        slack = cp.Parameter(1, nonneg=True)
        
        # Box constraints
        constraints = [yhat >= self.lb, yhat <= self.ub]
        
        if self.fairness_criterion == 'mean_residual':
            y = cp.Parameter(batch_size)
            
            # Add marginal fairness constraints for EACH protected attribute
            for attr_i in range(self.num_protected_attrs):
                selector_0 = selection_matrices[attr_i * 2]
                selector_1 = selection_matrices[attr_i * 2 + 1]
                
                n_0 = selector_0.sum()
                n_1 = selector_1.sum()
                
                # Skip if either group is empty
                if n_0 < 1 or n_1 < 1:
                    continue
                
                # Mean predictions for each group (using selection matrices)
                mean_pred_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
                mean_y_0 = cp.sum(cp.multiply(y, selector_0)) / n_0
                
                mean_pred_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
                mean_y_1 = cp.sum(cp.multiply(y, selector_1)) / n_1
                
                # Mean residual fairness: E[Y - Ŷ | A=a] ≈ 0 for each group
                constraints += [
                    mean_pred_0 - mean_y_0 <= slack,
                    mean_pred_0 - mean_y_0 >= -slack,
                    mean_pred_1 - mean_y_1 <= slack,
                    mean_pred_1 - mean_y_1 >= -slack,
                ]
            
            objective = cp.Minimize(cp.sum_squares(yhat - raw))
            problem = cp.Problem(objective, constraints)
            
            return CvxpyLayer(
                problem,
                parameters=[raw, y, slack],
                variables=[yhat]
            )
        else:
            # Mean prediction fairness
            # Add marginal fairness constraints for EACH protected attribute
            for attr_i in range(self.num_protected_attrs):
                selector_0 = selection_matrices[attr_i * 2]
                selector_1 = selection_matrices[attr_i * 2 + 1]
                
                n_0 = selector_0.sum()
                n_1 = selector_1.sum()
                
                # Skip if either group is empty
                if n_0 < 1 or n_1 < 1:
                    continue
                
                # Mean predictions for each group
                mean_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
                mean_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
                
                # Demographic parity: E[Ŷ | A=0] ≈ E[Ŷ | A=1]
                constraints += [
                    mean_0 - mean_1 <= slack,
                    mean_1 - mean_0 <= slack,
                ]
            
            objective = cp.Minimize(cp.sum_squares(yhat - raw))
            problem = cp.Problem(objective, constraints)
            
            return CvxpyLayer(
                problem,
                parameters=[raw, slack],
                variables=[yhat]
            )
    
    def _create_primal_dual_layer_marginal(
        self,
        batch_size: int,
        selection_matrices: List[np.ndarray],
        lambda_val: float
    ) -> CvxpyLayer:
        """
        Create cvxpylayers layer for primal-dual inference with MARGINAL fairness.
        
        The fairness constraints are moved to the objective as a penalty term.
        Uses max over all protected attributes for the gap penalty.
        
        Solves:
            minimize    ||yhat - raw||^2 + lambda * batch_size * max_gap
            subject to  lb <= yhat <= ub
        """
        # Single decision variable
        yhat = cp.Variable(batch_size)
        
        # Auxiliary variable for max gap across attributes
        max_gap = cp.Variable(1, nonneg=True)
        
        # Parameters
        raw = cp.Parameter(batch_size)
        
        # Box constraints
        constraints = [yhat >= self.lb, yhat <= self.ub]
        
        if self.fairness_criterion == 'mean_residual':
            y = cp.Parameter(batch_size)
            
            # Add constraints that max_gap >= gap for each attribute
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
                
                # max_gap >= |mean_residual| for each group
                residual_0 = mean_pred_0 - mean_y_0
                residual_1 = mean_pred_1 - mean_y_1
                
                constraints += [
                    max_gap >= residual_0,
                    max_gap >= -residual_0,
                    max_gap >= residual_1,
                    max_gap >= -residual_1,
                ]
            
            objective = cp.Minimize(
                cp.sum_squares(yhat - raw) + lambda_val * batch_size * max_gap
            )
            problem = cp.Problem(objective, constraints)
            
            return CvxpyLayer(
                problem,
                parameters=[raw, y],
                variables=[yhat]
            )
        else:
            # Mean prediction fairness
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
            
            objective = cp.Minimize(
                cp.sum_squares(yhat - raw) + lambda_val * batch_size * max_gap
            )
            problem = cp.Problem(objective, constraints)
            
            return CvxpyLayer(
                problem,
                parameters=[raw],
                variables=[yhat]
            )
    
    def _compute_batch_fairness_gap(
        self,
        ytilde: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute fairness gap for current batch.
        Returns max gap across ALL protected attributes.
        """
        max_gap = 0.0
        
        for attr_idx in self.protected_attr_idx:
            indicator_0 = x[:, attr_idx] == 0
            indicator_1 = ~indicator_0
            
            n_0 = indicator_0.sum().item()
            n_1 = indicator_1.sum().item()
            
            if n_0 == 0 or n_1 == 0:
                continue
            
            preds_0 = ytilde[indicator_0]
            preds_1 = ytilde[indicator_1]
            
            if self.fairness_criterion == 'mean_residual' and y is not None:
                y_squeezed = y.squeeze(-1) if y.dim() > 1 else y
                residual_0 = (y_squeezed[indicator_0] - preds_0.squeeze(-1)).mean().item()
                residual_1 = (y_squeezed[indicator_1] - preds_1.squeeze(-1)).mean().item()
                gap = max(abs(residual_0), abs(residual_1))
            else:
                mean_0 = preds_0.mean().item()
                mean_1 = preds_1.mean().item()
                gap = abs(mean_0 - mean_1)
            
            max_gap = max(max_gap, gap)
        
        return max_gap
    
    def _print_debug_info(
        self,
        context: str,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        selection_matrices: List[np.ndarray],
        error: Exception,
        lambda_val: float = None
    ):
        """Print detailed debug information when solver fails."""
        print("\n" + "="*80)
        print(f"CVXPY SOLVER FAILED - {context}")
        print("="*80)
        print(f"Error: {error}")
        print(f"\nBatch size: {len(x)}")
        print(f"b_tau threshold: {self.b_tau}")
        print(f"Fairness tolerance: {self.fairness_tolerance}")
        print(f"Number of protected attributes: {self.num_protected_attrs}")
        if lambda_val is not None:
            print(f"Lambda (dual variable): {lambda_val}")
        print(f"\nRaw predictions: min={y_hat.min().item():.4f}, max={y_hat.max().item():.4f}, mean={y_hat.mean().item():.4f}")
        
        for attr_i in range(self.num_protected_attrs):
            selector_0 = selection_matrices[attr_i * 2]
            selector_1 = selection_matrices[attr_i * 2 + 1]
            n_0 = selector_0.sum()
            n_1 = selector_1.sum()
            attr_idx = self.protected_attr_idx[attr_i]
            print(f"\nAttribute {attr_idx}: n_0={n_0:.0f}, n_1={n_1:.0f}")
        print("="*80 + "\n")
    
    def reset_inference_state(self):
        """
        Reset inference dual variables and statistics.
        Should be called before evaluation on a new dataset.
        """
        self.lambda_dual = 0.0
        self.dual_update_count = 0
        self.cumulative_samples = 0
        self.cumulative_weighted_violation = 0.0
        self.lambda_max = 0.0
    
    def get_aggregate_fairness_stats(
        self,
        loader,
        reset_before: bool = True
    ) -> dict:
        """
        Compute aggregate fairness statistics over a data loader.
        
        Runs inference on entire loader and computes:
        - Aggregate fairness gap per protected attribute
        - Max aggregate gap across all attributes
        - Theoretical bound from Theorem 2.2 (for small-batch inference)
        - Max dual variable seen
        
        Args:
            loader: DataLoader to evaluate
            reset_before: Whether to reset inference state before evaluation
            
        Returns:
            Dictionary with aggregate statistics
        """
        self.eval()
        
        if reset_before:
            self.reset_inference_state()
        
        # Collect predictions and targets by group for each attribute
        all_preds = {attr_idx: {0: [], 1: []} for attr_idx in self.protected_attr_idx}
        all_targets = {attr_idx: {0: [], 1: []} for attr_idx in self.protected_attr_idx}
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                # Skip batches without both groups in any attribute
                skip = False
                for attr_idx in self.protected_attr_idx:
                    if (batch_x[:, attr_idx] == 0).sum() < 1 or \
                       (batch_x[:, attr_idx] == 1).sum() < 1:
                        skip = True
                        break
                if skip:
                    continue
                
                # Get predictions (inference mode)
                if self.fairness_criterion == 'mean_residual':
                    preds = self(batch_x, y=batch_y, inference=True)
                else:
                    preds = self(batch_x, inference=True)
                
                # Collect by group for each attribute
                for attr_idx in self.protected_attr_idx:
                    indicator_0 = batch_x[:, attr_idx] == 0
                    
                    all_preds[attr_idx][0].extend(preds[indicator_0].squeeze(-1).cpu().tolist())
                    all_preds[attr_idx][1].extend(preds[~indicator_0].squeeze(-1).cpu().tolist())
                    
                    if batch_y is not None:
                        y_squeezed = batch_y.squeeze(-1) if batch_y.dim() > 1 else batch_y
                        all_targets[attr_idx][0].extend(y_squeezed[indicator_0].cpu().tolist())
                        all_targets[attr_idx][1].extend(y_squeezed[~indicator_0].cpu().tolist())
        
        # Compute aggregate statistics
        stats = {}
        max_gap = 0.0
        
        for attr_idx in self.protected_attr_idx:
            preds_0 = all_preds[attr_idx][0]
            preds_1 = all_preds[attr_idx][1]
            
            if len(preds_0) > 0 and len(preds_1) > 0:
                mean_pred_0 = np.mean(preds_0)
                mean_pred_1 = np.mean(preds_1)
                
                if self.fairness_criterion == 'mean_residual':
                    targets_0 = all_targets[attr_idx][0]
                    targets_1 = all_targets[attr_idx][1]
                    
                    if len(targets_0) > 0 and len(targets_1) > 0:
                        mean_residual_0 = np.mean(np.array(targets_0) - np.array(preds_0))
                        mean_residual_1 = np.mean(np.array(targets_1) - np.array(preds_1))
                        gap = max(abs(mean_residual_0), abs(mean_residual_1))
                        stats[f'mean_residual_attr_{attr_idx}_group_0'] = mean_residual_0
                        stats[f'mean_residual_attr_{attr_idx}_group_1'] = mean_residual_1
                    else:
                        gap = float('nan')
                else:
                    gap = abs(mean_pred_0 - mean_pred_1)
                
                stats[f'aggregate_gap_attr_{attr_idx}'] = gap
                stats[f'mean_pred_attr_{attr_idx}_group_0'] = mean_pred_0
                stats[f'mean_pred_attr_{attr_idx}_group_1'] = mean_pred_1
                
                if not np.isnan(gap):
                    max_gap = max(max_gap, gap)
        
        stats['aggregate_gap'] = max_gap
        stats['lambda_max'] = self.lambda_max
        stats['total_samples'] = self.cumulative_samples
        
        # Theoretical bound from Theorem 2.2
        if self.cumulative_samples > 0 and self.eta_0 > 0:
            eta_final = self.get_adaptive_eta()
            stats['theoretical_bound'] = self.fairness_tolerance + \
                (self.lambda_max / (eta_final * self.cumulative_samples))
        else:
            stats['theoretical_bound'] = float('inf')
        
        self.train()
        return stats