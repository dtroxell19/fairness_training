"""
Fair Neural Network with Differentiable Fairness Constraints

This module implements a neural network with an embedded cvxpylayers optimization 
layer that enforces group fairness constraints during training through differentiable
convex optimization.

Key Features:
- Differentiable fairness layer using cvxpylayers
- Sliding window of past predictions (P_w) for constraint stabilization
- Configurable slack schedule for gradual constraint tightening
- Separate handling of training and inference modes
- Support for multiple protected attributes (marginal fairness)
- Support for both mean prediction and mean residual fairness
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
    cvxpylayers optimization layer that enforces demographic parity constraints
    while minimizing distortion from the original predictions.
    
    Args:
        input_dim (int): Number of input features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Output dimension (typically 1 for binary classification)
        protected_attr_idx (Union[int, List[int]]): Index or list of indices of protected attribute columns
        prediction_bounds (Tuple[float, float]): (lower, upper) bounds for predictions
        initial_slack (float): Initial slack tolerance for fairness constraint
        min_slack (float): Minimum slack value (for decay schedule)
        slack_decay (float): Decay factor for slack per epoch (default: 0.99)
        activation (str): Activation function ('relu', 'tanh', 'sigmoid')
        b_tau (int): Size of sliding window for past predictions
        fairness_criterion (str): 'mean_pred' for equal mean predictions, 
                                   'mean_residual' for equal mean residuals
        
    Example:
        # Single protected attribute (backward compatible)
        model = FairModel(input_dim=20, hidden_dims=[64,32], protected_attr_idx=0)
        
        # Multiple protected attributes
        model = FairModel(input_dim=20, hidden_dims=[64,32], protected_attr_idx=[0, 1, 3])
        
        # Mean residual fairness (for regression)
        model = FairModel(input_dim=20, hidden_dims=[64,32], 
                         protected_attr_idx=0, fairness_criterion='mean_residual')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        protected_attr_idx: Union[int, List[int]] = 0,
        prediction_bounds: Tuple[float, float] = (0.0, 1.0),
        initial_slack: float = 0.05,
        min_slack: float = 0.001,
        slack_decay: float = 0.999,
        activation: str = 'relu',
        b_tau: int = 1000,
        fairness_criterion: str = 'mean_pred',
        train_batch_size: int = 256,
        eval_batch_size: int = 256,
        warmup_batches_threshold: int = 0,
        warmup_inference_batches: int = 100    # Inference warmup
    ):
        super(FairModel, self).__init__()
        
        self.warmup_train_batches = 0
        self.warmup_train_threshold = warmup_batches_threshold
        self.warmup_inference_batches = 0
        self.warmup_inference_threshold = warmup_inference_batches
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.last_solve_predictions = None
        self.last_solve_indicators = None  # Dict mapping attr_idx to indicators
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
        self.initial_slack = initial_slack
        self.min_slack = min_slack
        self.slack_decay = slack_decay
        self.slack_current = torch.tensor([initial_slack], dtype=torch.float32)
        
        # Sliding window parameters (for stabilization)
        self.P_w = None  # Training window
        self.indicator_past = None  # Group indicators for training window (list of arrays)
        self.P_inf = None  # Inference window
        self.indicator_inf = None  # Group indicators for inference window
        self.b_tau = b_tau
        
        # Build the feedforward network
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
            
    def forward(
        self, 
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        inference: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with fairness projection.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            y: Target tensor (required if fairness_criterion='mean_residual')
            inference: If True, use inference mode with separate sliding window
            
        Returns:
            Fair predictions of shape (batch_size, output_dim)
        """
        # Get raw predictions from neural network
        y_hat = self.ffnn(x)
        
        # Check if targets are needed
        if self.fairness_criterion == 'mean_residual' and y is None:
            raise ValueError("Target y must be provided when fairness_criterion='mean_residual'")
        
        # Warmup period: return raw predictions BUT still update windows
        if inference:
            if self.warmup_inference_batches < self.warmup_inference_threshold:
                self.warmup_inference_batches += 1
                # Update inference window with raw predictions
                batch_size = len(x)
                if batch_size < self.b_tau:
                    if self.P_inf is None:
                        self.P_inf = np.array([])
                        self.indicator_inf = [[] for _ in range(self.num_protected_attrs)]
                    self._update_inference_window(y_hat, x)
                return y_hat
        else:
            if self.warmup_train_batches < self.warmup_train_threshold:
                self.warmup_train_batches += 1
                # Update training window with raw predictions
                batch_size = len(x)
                if batch_size < self.b_tau:
                    if self.P_w is None:
                        self.P_w = np.array([])
                        self.indicator_past = [[] for _ in range(self.num_protected_attrs)]
                    self._update_training_window(y_hat, x)           
                return y_hat
        
        # Ensure slack is tensor with correct dtype
        if not isinstance(self.slack_current, torch.Tensor):
            self.slack_current = torch.tensor([float(self.slack_current)], dtype=torch.float32)
        elif self.slack_current.dtype != torch.float32:
            self.slack_current = self.slack_current.to(torch.float32)
        
        # Apply fairness projection (after warmup)
        if inference:
            ytilde = self._forward_inference(x, y_hat, y)
        else:
            ytilde = self._forward_training(x, y_hat, y)
            
        return ytilde
            
    def _forward_training(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Training forward pass with sliding window stabilization."""
        batch_size = len(x)
        use_window = batch_size < self.b_tau and self.P_w is not None and self.indicator_past is not None and len(self.P_w) > 0
        
        # Create selection matrices for each protected attribute
        selection_matrices_current = self._create_selection_matrices(x)
        
        # Create fairness layer
        if use_window:
            # Combine current batch with past predictions from window
            selection_matrices_combined = self._combine_with_window(
                selection_matrices_current, self.indicator_past, len(self.P_w)
            )
            
            self.cvxpylayer = self._create_fair_layer(
                batch_size + len(self.P_w),
                selection_matrices_combined,
                use_past=True,
                past_size=len(self.P_w)
            )
            
            # Prepare past predictions as fixed parameters
            yhat_past = torch.from_numpy(self.P_w).to(torch.float32)
            
            try:
                if self.fairness_criterion == 'mean_residual':
                    y_past = torch.from_numpy(self.P_w).to(torch.float32)
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1),
                        y.squeeze(1),
                        self.slack_current,
                        yhat_past,
                        y_past
                    )[0]
                else:
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1),
                        self.slack_current,
                        yhat_past
                    )[0]
            except Exception as e:
                print("\n" + "="*80)
                print("CVXPY SOLVER FAILED (with window)")
                print("="*80)
                print(f"Error: {e}")
                print(f"\nProblem dimensions:")
                print(f"  Current batch size: {batch_size}")
                print(f"  Past window size: {len(self.P_w)}")
                print(f"  Total size: {batch_size + len(self.P_w)}")
                print(f"\nSlack: {self.slack_current.item()}")
                print(f"\nRaw predictions (current):")
                print(f"  Shape: {y_hat.squeeze(1).shape}")
                print(f"  Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}")
                print(f"  Mean: {y_hat.mean().item():.4f}, Std: {y_hat.std().item():.4f}")
                print(f"\nPast predictions:")
                print(f"  Shape: {yhat_past.shape}")
                print(f"  Min: {yhat_past.min().item():.4f}, Max: {yhat_past.max().item():.4f}")
                print(f"  Mean: {yhat_past.mean().item():.4f}, Std: {yhat_past.std().item():.4f}")
                
                print(f"\n--- RAW PREDICTIONS (current batch only) ---")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    selector_0 = selection_matrices_current[attr_i * 2]
                    selector_1 = selection_matrices_current[attr_i * 2 + 1]
                    n_0_raw = selector_0.sum()
                    n_1_raw = selector_1.sum()
                    
                    mask_0 = torch.from_numpy(selector_0.astype(bool))
                    mask_1 = torch.from_numpy(selector_1.astype(bool))
                    mean_0_raw = y_hat.squeeze(1)[mask_0].mean().item() if n_0_raw > 0 else 0
                    mean_1_raw = y_hat.squeeze(1)[mask_1].mean().item() if n_1_raw > 0 else 0
                    gap_raw = abs(mean_0_raw - mean_1_raw)
                    
                    print(f"  Attribute {attr_idx}: n_0={n_0_raw:.0f}, n_1={n_1_raw:.0f} | mean_0={mean_0_raw:.4f}, mean_1={mean_1_raw:.4f} | gap={gap_raw:.6f}")
                
                print(f"\n--- PAST PREDICTIONS (window) ---")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    past_indicator = np.array(self.indicator_past[attr_i])
                    past_mask_0 = (past_indicator == 0)
                    past_mask_1 = (past_indicator == 1)
                    n_0_past = past_mask_0.sum()
                    n_1_past = past_mask_1.sum()
                    
                    mean_0_past = yhat_past[torch.from_numpy(past_mask_0)].mean().item() if n_0_past > 0 else 0
                    mean_1_past = yhat_past[torch.from_numpy(past_mask_1)].mean().item() if n_1_past > 0 else 0
                    gap_past = abs(mean_0_past - mean_1_past)
                    
                    print(f"  Attribute {attr_idx}: n_0={n_0_past:.0f}, n_1={n_1_past:.0f} | mean_0={mean_0_past:.4f}, mean_1={mean_1_past:.4f} | gap={gap_past:.6f}")
                
                print(f"\n--- COMBINED (current + past) ---")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    selector_0 = selection_matrices_combined[attr_i * 2]
                    selector_1 = selection_matrices_combined[attr_i * 2 + 1]
                    n_0_total = selector_0.sum()
                    n_1_total = selector_1.sum()
                    
                    # Compute what the gap would be without fairness constraint
                    all_preds = torch.cat([y_hat.squeeze(1), yhat_past])
                    mean_0 = all_preds[torch.from_numpy(selector_0.astype(bool))].mean().item()
                    mean_1 = all_preds[torch.from_numpy(selector_1.astype(bool))].mean().item()
                    gap = abs(mean_0 - mean_1)
                    print(f"  Attribute {attr_idx}: n_0={n_0_total:.0f}, n_1={n_1_total:.0f} | mean_0={mean_0:.4f}, mean_1={mean_1:.4f} | gap={gap:.6f} (slack: {self.slack_current.item():.6f})")
                print("="*80 + "\n")
                raise
            
            ytilde = ytilde.unsqueeze(1)
            
            # SAVE what was actually used in this solve (BEFORE updating window)
            self.last_solve_predictions = torch.cat([ytilde.detach(), yhat_past.unsqueeze(1)], dim=0)
            self.last_solve_indicators = {}
            for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                current_indicators = x[:, attr_idx].cpu().numpy()
                past_indicators_array = np.array(self.indicator_past[attr_i])
                combined_indicators = np.concatenate([current_indicators, past_indicators_array])
                self.last_solve_indicators[attr_idx] = combined_indicators
            
        else:
            # First batch or large batch - no past predictions for fairness constraints
            self.cvxpylayer = self._create_fair_layer(batch_size, selection_matrices_current)
            
            try:
                if self.fairness_criterion == 'mean_residual':
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1), 
                        y.squeeze(1), 
                        self.slack_current
                    )[0]
                else:
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1), 
                        self.slack_current
                    )[0]
            except Exception as e:
                print("\n" + "="*80)
                print("CVXPY SOLVER FAILED (no window)")
                print("="*80)
                print(f"Error: {e}")
                print(f"\nProblem dimensions:")
                print(f"  Batch size: {batch_size}")
                print(f"\nSlack: {self.slack_current.item()}")
                print(f"\nRaw predictions:")
                print(f"  Shape: {y_hat.squeeze(1).shape}")
                print(f"  Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}")
                print(f"  Mean: {y_hat.mean().item():.4f}, Std: {y_hat.std().item():.4f}")
                print(f"\nGroup distributions:")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    selector_0 = selection_matrices_current[attr_i * 2]
                    selector_1 = selection_matrices_current[attr_i * 2 + 1]
                    print(f"  Attribute {attr_idx}: Group 0 = {selector_0.sum():.0f}, Group 1 = {selector_1.sum():.0f}")
                    
                    # Compute what the gap would be without fairness constraint
                    mask_0 = torch.from_numpy(selector_0.astype(bool))
                    mask_1 = torch.from_numpy(selector_1.astype(bool))
                    mean_0 = y_hat.squeeze(1)[mask_0].mean().item()
                    mean_1 = y_hat.squeeze(1)[mask_1].mean().item()
                    gap = abs(mean_0 - mean_1)
                    print(f"    Unconstrained gap: {gap:.6f} (slack: {self.slack_current.item():.6f})")
                print("="*80 + "\n")
                raise
            
            ytilde = ytilde.unsqueeze(1)
            
            # SAVE what was actually used (just current batch)
            self.last_solve_predictions = ytilde.detach()
            self.last_solve_indicators = {}
            for attr_idx in self.protected_attr_idx:
                self.last_solve_indicators[attr_idx] = x[:, attr_idx].cpu().numpy()
        
        # ALWAYS update training window (even for large batches) to seed inference
        # Initialize window if it doesn't exist
        if self.P_w is None:
            self.P_w = np.array([])
            self.indicator_past = [[] for _ in range(self.num_protected_attrs)]
        
        self._update_training_window(ytilde, x)
        
        # Update slack
        if batch_size < self.b_tau:
            new_slack = float(max(
                self.slack_current.item() * self.slack_decay,
                self.min_slack
            ))
            self.slack_current = torch.tensor([new_slack], dtype=torch.float32)
        else:
            self.slack_current = torch.tensor([self.min_slack], dtype=torch.float32)
        
        return ytilde
            
    def _forward_inference(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference forward pass with separate sliding window."""
        batch_size = len(x)
        
        # Initialize inference window from training window if available
        if (self.P_inf is None or len(self.P_inf) == 0) and self.P_w is not None and len(self.P_w) > 0:
            print(f"📋 Initializing inference window from training window ({len(self.P_w)} predictions)")
            self.P_inf = self.P_w.copy()  # Copy training predictions
            self.indicator_inf = [indicators.copy() for indicators in self.indicator_past]  # Copy indicators
        
        use_window = (batch_size < self.b_tau and 
                    self.P_inf is not None and 
                    self.indicator_inf is not None and
                    len(self.P_inf) > 0)
        
        selection_matrices_current = self._create_selection_matrices(x)
        min_slack_tensor = torch.tensor([self.min_slack], dtype=torch.float32)
        
        if use_window:
            P_inf_tensor = torch.from_numpy(np.array(self.P_inf)).to(torch.float32)
            
            selection_matrices_combined = self._combine_with_window(
                selection_matrices_current, self.indicator_inf, len(P_inf_tensor)
            )
            
            self.cvxpylayer = self._create_fair_layer(
                batch_size + len(P_inf_tensor),
                selection_matrices_combined,
                use_past=True,
                past_size=len(P_inf_tensor)
            )
            
            yhat_past = P_inf_tensor
            
            try:
                if self.fairness_criterion == 'mean_residual':
                    y_past = P_inf_tensor
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1),
                        y.squeeze(1),
                        min_slack_tensor,
                        yhat_past,
                        y_past
                    )[0]
                else:
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1),
                        min_slack_tensor,
                        yhat_past
                    )[0]

            except Exception as e:
                print("\n" + "="*80)
                print("CVXPY SOLVER FAILED - INFERENCE (with window)")
                print("="*80)
                print(f"Error: {e}")
                print(f"\nProblem dimensions:")
                print(f"  Current batch size: {batch_size}")
                print(f"  Past window size: {len(P_inf_tensor)}")
                print(f"  Total size: {batch_size + len(P_inf_tensor)}")
                print(f"\nSlack: {min_slack_tensor.item()}")
                print(f"\nRaw predictions (current):")
                print(f"  Shape: {y_hat.squeeze(1).shape}")
                print(f"  Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}")
                print(f"  Mean: {y_hat.mean().item():.4f}, Std: {y_hat.std().item():.4f}")
                print(f"\nPast predictions:")
                print(f"  Shape: {yhat_past.shape}")
                print(f"  Min: {yhat_past.min().item():.4f}, Max: {yhat_past.max().item():.4f}")
                print(f"  Mean: {yhat_past.mean().item():.4f}, Std: {yhat_past.std().item():.4f}")
                
                print(f"\n--- RAW PREDICTIONS (current batch only) ---")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    selector_0 = selection_matrices_current[attr_i * 2]
                    selector_1 = selection_matrices_current[attr_i * 2 + 1]
                    n_0_raw = selector_0.sum()
                    n_1_raw = selector_1.sum()
                    
                    mask_0 = torch.from_numpy(selector_0.astype(bool))
                    mask_1 = torch.from_numpy(selector_1.astype(bool))
                    mean_0_raw = y_hat.squeeze(1)[mask_0].mean().item() if n_0_raw > 0 else 0
                    mean_1_raw = y_hat.squeeze(1)[mask_1].mean().item() if n_1_raw > 0 else 0
                    gap_raw = abs(mean_0_raw - mean_1_raw)
                    
                    print(f"  Attribute {attr_idx}: n_0={n_0_raw:.0f}, n_1={n_1_raw:.0f} | mean_0={mean_0_raw:.4f}, mean_1={mean_1_raw:.4f} | gap={gap_raw:.6f}")
                
                print(f"\n--- PAST PREDICTIONS (window) ---")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    past_indicator = np.array(self.indicator_inf[attr_i])
                    past_mask_0 = (past_indicator == 0)
                    past_mask_1 = (past_indicator == 1)
                    n_0_past = past_mask_0.sum()
                    n_1_past = past_mask_1.sum()
                    
                    mean_0_past = yhat_past[torch.from_numpy(past_mask_0)].mean().item() if n_0_past > 0 else 0
                    mean_1_past = yhat_past[torch.from_numpy(past_mask_1)].mean().item() if n_1_past > 0 else 0
                    gap_past = abs(mean_0_past - mean_1_past)
                    
                    print(f"  Attribute {attr_idx}: n_0={n_0_past:.0f}, n_1={n_1_past:.0f} | mean_0={mean_0_past:.4f}, mean_1={mean_1_past:.4f} | gap={gap_past:.6f}")
                
                print(f"\n--- COMBINED (current + past) ---")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    selector_0 = selection_matrices_combined[attr_i * 2]
                    selector_1 = selection_matrices_combined[attr_i * 2 + 1]
                    n_0_total = selector_0.sum()
                    n_1_total = selector_1.sum()
                    
                    # Compute what the gap would be without fairness constraint
                    all_preds = torch.cat([y_hat.squeeze(1), yhat_past])
                    mean_0 = all_preds[torch.from_numpy(selector_0.astype(bool))].mean().item()
                    mean_1 = all_preds[torch.from_numpy(selector_1.astype(bool))].mean().item()
                    gap = abs(mean_0 - mean_1)
                    print(f"  Attribute {attr_idx}: n_0={n_0_total:.0f}, n_1={n_1_total:.0f} | mean_0={mean_0:.4f}, mean_1={mean_1:.4f} | gap={gap:.6f} (slack: {min_slack_tensor.item():.6f})")
                print("="*80 + "\n")
                raise
            ytilde = ytilde.unsqueeze(1)
            
            # SAVE what was actually used in this solve
            self.last_solve_predictions = torch.cat([ytilde.detach(), yhat_past.unsqueeze(1)], dim=0)
            self.last_solve_indicators = {}
            for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                current_indicators = x[:, attr_idx].cpu().numpy()
                past_indicators_array = np.array(self.indicator_inf[attr_i])
                combined_indicators = np.concatenate([current_indicators, past_indicators_array])
                self.last_solve_indicators[attr_idx] = combined_indicators
            
        else:
            self.cvxpylayer = self._create_fair_layer(batch_size, selection_matrices_current)
            
            try:
                if self.fairness_criterion == 'mean_residual':
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1), 
                        y.squeeze(1), 
                        min_slack_tensor
                    )[0]
                else:
                    ytilde = self.cvxpylayer(
                        y_hat.squeeze(1), 
                        min_slack_tensor
                    )[0]
            except Exception as e:
                print("\n" + "="*80)
                print("CVXPY SOLVER FAILED - INFERENCE (no window)")
                print("="*80)
                print(f"Error: {e}")
                print(f"\nProblem dimensions:")
                print(f"  Batch size: {batch_size}")
                print(f"\nSlack: {min_slack_tensor.item()}")
                print(f"\nRaw predictions:")
                print(f"  Shape: {y_hat.squeeze(1).shape}")
                print(f"  Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}")
                print(f"  Mean: {y_hat.mean().item():.4f}, Std: {y_hat.std().item():.4f}")
                print(f"\nGroup distributions:")
                for attr_i, attr_idx in enumerate(self.protected_attr_idx):
                    selector_0 = selection_matrices_current[attr_i * 2]
                    selector_1 = selection_matrices_current[attr_i * 2 + 1]
                    print(f"  Attribute {attr_idx}: Group 0 = {selector_0.sum():.0f}, Group 1 = {selector_1.sum():.0f}")
                    
                    # Compute what the gap would be without fairness constraint
                    mask_0 = torch.from_numpy(selector_0.astype(bool))
                    mask_1 = torch.from_numpy(selector_1.astype(bool))
                    mean_0 = y_hat.squeeze(1)[mask_0].mean().item()
                    mean_1 = y_hat.squeeze(1)[mask_1].mean().item()
                    gap = abs(mean_0 - mean_1)
                    print(f"    Unconstrained gap: {gap:.6f} (slack: {min_slack_tensor.item():.6f})")
                print("="*80 + "\n")
                raise
            
            ytilde = ytilde.unsqueeze(1)
            
            # SAVE what was actually used
            self.last_solve_predictions = ytilde.detach()
            self.last_solve_indicators = {}
            for attr_idx in self.protected_attr_idx:
                self.last_solve_indicators[attr_idx] = x[:, attr_idx].cpu().numpy()
            
            if batch_size < self.b_tau:
                self.P_inf = np.array([])  # <- Use numpy like training
                self.indicator_inf = [[] for _ in range(self.num_protected_attrs)]
        
        # Update inference window
        if batch_size < self.b_tau:
            self._update_inference_window(ytilde, x)
        
        return ytilde

    def _create_selection_matrices(self, x: torch.Tensor) -> List[np.ndarray]:
        """
        Create binary selection matrices for each protected attribute and group.
        
        Returns list of 2*num_protected_attrs numpy arrays of shape (batch_size,)
        with 0/1 entries. For each attribute i: [selector_group0, selector_group1]
        """
        batch_size = len(x)
        selection_matrices = []
        
        for attr_idx in self.protected_attr_idx:
            protected_vals = x[:, attr_idx].cpu().numpy()
            
            # Group 0 selector
            selector_0 = (protected_vals == 0).astype(np.float32)
            # Group 1 selector
            selector_1 = (protected_vals == 1).astype(np.float32)
            
            selection_matrices.append(selector_0)
            selection_matrices.append(selector_1)
        
        return selection_matrices
    
    def _combine_with_window(
        self, 
        current_selectors: List[np.ndarray], 
        past_indicators: List[List[int]], 
        past_size: int
    ) -> List[np.ndarray]:
        """
        Combine current selection matrices with past indicators from sliding window.
        
        Args:
            current_selectors: Selection matrices for current batch
            past_indicators: List of lists containing past group indicators for each attribute
            past_size: Number of past predictions
            
        Returns:
            Combined selection matrices of shape (batch_size + past_size,)
        """
        combined = []
        
        for attr_i in range(self.num_protected_attrs):
            # Current batch selectors
            current_0 = current_selectors[attr_i * 2]
            current_1 = current_selectors[attr_i * 2 + 1]
            
            # Past indicators for this attribute
            past_indicator = np.array(past_indicators[attr_i])
            past_0 = (past_indicator == 0).astype(np.float32)
            past_1 = (past_indicator == 1).astype(np.float32)
            
            # Concatenate
            combined_0 = np.concatenate([current_0, past_0])
            combined_1 = np.concatenate([current_1, past_1])
            
            combined.append(combined_0)
            combined.append(combined_1)
        
        return combined
    
    def _create_fair_layer(
        self,
        batch_size: int,
        selection_matrices: List[np.ndarray],
        use_past: bool = False,
        past_size: int = 0
    ) -> CvxpyLayer:
        """
        Create cvxpylayers optimization layer for fairness projection with multiple protected attributes.
        
        The optimization problem:
            minimize    ||ytilde_current - raw_current||^2
            subject to  lb <= ytilde <= ub (for all, including past)
                        For each protected attribute:
                            |mean(ytilde_all | group0) - mean(y_all | group0)| <= slack  (if mean_residual)
                            |mean(ytilde_all | group1) - mean(y_all | group1)| <= slack  (if mean_residual)
                        OR
                            |mean(ytilde_all | group0) - mean(ytilde_all | group1)| <= slack (if mean_pred)
        
        When use_past=True:
            - batch_size includes current batch + past predictions
            - Only optimize current predictions (minimize distortion for current only)
            - But fairness constraints use all predictions (current + past) for stability
        """
        if use_past:
            # Total size = current batch + past
            # Decision variables for current batch only
            current_size = batch_size - past_size
            yhat_current = cp.Variable(current_size)
            
            # Past predictions are fixed (parameters, not variables)
            yhat_past = cp.Parameter(past_size)
            
            # Concatenate for constraint computation
            yhat_all = cp.hstack([yhat_current, yhat_past])
        else:
            # Simple case: only current batch
            yhat_current = cp.Variable(batch_size)
            yhat_all = yhat_current
        
        # Parameters
        raw = cp.Parameter(batch_size if not use_past else batch_size - past_size)
        slack = cp.Parameter(1)
        
        if self.fairness_criterion == 'mean_residual':
            if use_past:
                y_current = cp.Parameter(batch_size - past_size)
                y_past = cp.Parameter(past_size)
                y_all = cp.hstack([y_current, y_past])
            else:
                y_all = cp.Parameter(batch_size)
        
        # Constraints: box bounds
        constr = [yhat_current >= self.lb, yhat_current <= self.ub]
        
        # Marginal fairness constraints for each protected attribute
        # These use yhat_all (current + past) for stability
        for attr_i in range(self.num_protected_attrs):
            selector_0 = selection_matrices[attr_i * 2]
            selector_1 = selection_matrices[attr_i * 2 + 1]
            
            n_0 = selector_0.sum()
            n_1 = selector_1.sum()
            
            # Skip if a group is empty
            if n_0 < 1 or n_1 < 1:
                continue
            
            if self.fairness_criterion == 'mean_residual':
                # Mean residual fairness: E[Y - Ŷ | A=a] should be close to 0 for all groups
                # Uses ALL predictions (current + past) for stable constraints
                mean_pred_0 = cp.sum(cp.multiply(yhat_all, selector_0)) / n_0
                mean_y_0 = cp.sum(cp.multiply(y_all, selector_0)) / n_0
                
                mean_pred_1 = cp.sum(cp.multiply(yhat_all, selector_1)) / n_1
                mean_y_1 = cp.sum(cp.multiply(y_all, selector_1)) / n_1
                
                # Residual constraints: mean(Y - Ŷ) ≈ 0 for each group
                constr += [
                    mean_pred_0 - mean_y_0 <= slack,
                    mean_pred_0 - mean_y_0 >= -slack,
                    mean_pred_1 - mean_y_1 <= slack,
                    mean_pred_1 - mean_y_1 >= -slack,
                ]
            else:
                # Mean prediction fairness: E[Ŷ | A=0] ≈ E[Ŷ | A=1]
                # Uses ALL predictions (current + past) for stable constraints
                mean_pred_0 = cp.sum(cp.multiply(yhat_all, selector_0)) / n_0
                mean_pred_1 = cp.sum(cp.multiply(yhat_all, selector_1)) / n_1
                
                # Demographic parity constraints
                constr += [
                    mean_pred_0 - mean_pred_1 <= slack,
                    mean_pred_1 - mean_pred_0 <= slack,
                ]
        
        # Objective: minimize distortion from raw predictions (ONLY for current batch)
        objective = cp.Minimize(cp.sum_squares(yhat_current - raw))
        
        problem = cp.Problem(objective, constr)
        
        # Create cvxpylayer with appropriate parameters
        if use_past:
            if self.fairness_criterion == 'mean_residual':
                return CvxpyLayer(
                    problem,
                    parameters=[raw, y_current, slack, yhat_past, y_past],
                    variables=[yhat_current]
                )
            else:
                return CvxpyLayer(
                    problem,
                    parameters=[raw, slack, yhat_past],
                    variables=[yhat_current]
                )
        else:
            if self.fairness_criterion == 'mean_residual':
                return CvxpyLayer(
                    problem,
                    parameters=[raw, y_all, slack],
                    variables=[yhat_current]
                )
            else:
                return CvxpyLayer(
                    problem,
                    parameters=[raw, slack],
                    variables=[yhat_current]
                )
    
    def _update_training_window(self, ytilde: torch.Tensor, x: torch.Tensor):
        """Update the sliding window of past training predictions."""
        # Ensure P_w is 1D
        if self.P_w.ndim > 1:
            self.P_w = self.P_w.reshape(-1)
        
        # Add new predictions
        new_preds = ytilde.squeeze(1).detach().cpu().numpy()
        self.P_w = np.concatenate([self.P_w, new_preds])
        
        # Add group indicators for each protected attribute
        for attr_i, attr_idx in enumerate(self.protected_attr_idx):
            protected_vals = x[:, attr_idx].cpu().numpy()
            indicators = (protected_vals == 0).astype(int).tolist()
            self.indicator_past[attr_i].extend(indicators)
        
        # Trim to window size
        if len(self.P_w) > self.b_tau-self.train_batch_size:
            self.P_w = self.P_w[-(self.b_tau-self.train_batch_size):]
            for attr_i in range(self.num_protected_attrs):
                self.indicator_past[attr_i] = self.indicator_past[attr_i][-(self.b_tau-self.train_batch_size):]
    
    def _update_inference_window(self, ytilde: torch.Tensor, x: torch.Tensor):
        """Update the sliding window of past inference predictions."""
        # Ensure P_inf is 1D
        if self.P_inf.ndim > 1:
            self.P_inf = self.P_inf.reshape(-1)
        
        # Add new predictions (convert to numpy)
        new_preds = ytilde.squeeze(-1).detach().cpu().numpy()
        self.P_inf = np.concatenate([self.P_inf, new_preds])  # Use np.concatenate, not extend
        
        # Add group indicators for each protected attribute
        for attr_i, attr_idx in enumerate(self.protected_attr_idx):
            protected_vals = x[:, attr_idx].cpu().numpy()
            indicators = (protected_vals == 0).astype(int).tolist()
            self.indicator_inf[attr_i].extend(indicators)  # Keep as list
        
        # Trim to window size
        if len(self.P_inf) > self.b_tau - self.eval_batch_size:
            self.P_inf = self.P_inf[-(self.b_tau - self.eval_batch_size):]
            for attr_i in range(self.num_protected_attrs):
                self.indicator_inf[attr_i] = self.indicator_inf[attr_i][-(self.b_tau - self.eval_batch_size):]
    
    def reset_windows(self, mode: str = 'both'):
        """
        Reset sliding windows.
        
        Args:
            mode: 'train', 'inference', or 'both'
        """
        if mode in ['train', 'both']:
            self.P_w = None
            self.indicator_past = None
        if mode in ['inference', 'both']:
            self.P_inf = None
            self.indicator_inf = None
    
    def reset_slack(self):
        """Reset slack to initial value."""
        self.slack_current = torch.tensor([self.initial_slack], dtype=torch.float32)
