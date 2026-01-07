"""
Example: Fair Neural Networks with Differentiable Fairness Constraints

This module demonstrates how to use the Fair Neural Networks package with the
primal-dual algorithm for enforcing fairness constraints during inference.

Key Insight: Training always uses large batches with hard per-batch constraints.
             Inference can use either:
             - Large batches: Hard per-batch constraints
             - Small batches: Online primal-dual algorithm (Algorithm 1) with 
                              provable aggregate fairness (Theorem 2.2)

Examples included:
1. Large-batch inference: Hard constraints for both training and inference
2. Small-batch inference: Hard constraints for training, primal-dual for inference
3. Custom network architecture support
4. Mean residual fairness for regression tasks

Dataset: Adult Income (UCI Machine Learning Repository)
Task: Predict whether income exceeds $50K/year
Protected Attributes: Gender and Race (white vs non-white)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from fair_model import FairModel
from trainer import FairTrainer, create_dataloaders, create_stratified_dataloaders
import utils


def load_and_preprocess_adult_data():
    """
    Load and preprocess the Adult Income dataset.
    
    Returns:
        X: Feature matrix with protected attributes as columns 0 and 1
        y: Target (income > 50K)
        feature_names: List of feature names
    """
    # Column names for Adult dataset
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    # Load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    try:
        df = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)
        print("Successfully loaded Adult Income dataset from UCI")
    except Exception as e:
        print(f"✗ Could not download dataset: {e}")
        return None, None, None
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Create binary target: 1 if income > 50K, 0 otherwise
    df['income_binary'] = (df['income'] == '>50K').astype(int)
    
    # Create protected attributes
    # Protected attribute 1: Gender (0 = Female, 1 = Male)
    df['gender_binary'] = (df['sex'] == 'Male').astype(int)
    
    # Protected attribute 2: Race (0 = Non-White, 1 = White)
    df['race_binary'] = (df['race'] == 'White').astype(int)
    
    # One-hot encode categorical features
    categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'native_country']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Get all feature columns (including one-hot encoded)
    all_feature_cols = [col for col in df_encoded.columns 
                        if col not in ['income', 'income_binary', 'gender_binary', 'race_binary', 
                                      'sex', 'race', 'education']]
    
    # Create feature matrix
    X_features = df_encoded[all_feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)
    
    # Add protected attributes as first two columns
    gender = df_encoded['gender_binary'].values.reshape(-1, 1).astype(np.float32)
    race = df_encoded['race_binary'].values.reshape(-1, 1).astype(np.float32)
    
    X = np.hstack([gender, race, X_features_scaled])
    y = df_encoded['income_binary'].values.astype(np.float32)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Protected attributes: Gender (col 0), Race (col 1)")
    print(f"\nGender distribution: Female={100*(1-gender.mean()):.1f}%, Male={100*gender.mean():.1f}%")
    print(f"Race distribution: Non-White={100*(1-race.mean()):.1f}%, White={100*race.mean():.1f}%")
    print(f"Income >50K: {100*y.mean():.1f}%")
    
    feature_names = ['gender', 'race'] + all_feature_cols
    
    return X, y, feature_names


def example_large_batch_inference():
    """
    Example 1: Large-Batch Inference (Hard Per-Batch Constraints Throughout)
    
    Both training and inference use large batches with hard per-batch fairness
    constraints. This is the simplest case and provides exact per-batch guarantees.
    
    Key settings:
    - Training batch_size = 2000 >= b_tau = 2000: Hard constraints
    - Inference batch_size = 2000 >= b_tau = 2000: Hard constraints
    - Uses create_stratified_dataloaders() to maintain constant group ratios
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: LARGE-BATCH INFERENCE (Hard Constraints Throughout)")
    print("="*80)
    
    # Load data
    X, y, feature_names = load_and_preprocess_adult_data()
    if X is None:
        return None
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Configuration - LARGE batch sizes for both training and inference
    batch_size_train = 2000  # Large batch for training
    batch_size_eval = 2000   # Large batch for inference
    b_tau = 2000             # Threshold for hard constraints
    fairness_tolerance = 0.05
    
    # Create STRATIFIED dataloaders (maintains constant group ratios per Lemma 2.1)
    train_loader, val_loader, test_loader = create_stratified_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        protected_attr_idx=[0,1],  # Stratify on gender
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval
    )
    
    # Create model
    input_dim = X.shape[1]
    model = FairModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        protected_attr_idx=[0, 1],  # Gender and Race
        prediction_bounds=(-10.0, 10.0),
        fairness_tolerance=fairness_tolerance,
        b_tau=b_tau,
        fairness_criterion='mean_pred'
    )
    
    print(f"\nModel configuration:")
    print(f"  - Training:  batch_size={batch_size_train} >= b_tau={b_tau} -> HARD constraints")
    print(f"  - Inference: batch_size={batch_size_eval} >= b_tau={b_tau} -> HARD constraints")
    print(f"  - fairness_tolerance (epsilon)={fairness_tolerance}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = FairTrainer(
        model, criterion, optimizer, 
        scheduler=scheduler,
        early_stopping_patience=15
    )
    
    # Train
    print("\n--- Training ---")
    history = trainer.fit(
        train_loader, val_loader, 
        epochs=50, 
        verbose=1, 
        log_interval=5
    )
    
    # Evaluate
    print("\n--- Test Evaluation ---")
    metrics = trainer.evaluate(test_loader, return_predictions=True)
    
    print(f"\nResults (Large-Batch Inference):")
    print(f"  Test Loss: {metrics['test_loss']:.4f}")
    print(f"  Fairness Gap (Gender): {metrics['fairness_gap_attr_0']:.4f}")
    print(f"  Fairness Gap (Race): {metrics['fairness_gap_attr_1']:.4f}")
    print(f"  Max Fairness Gap: {metrics['fairness_gap']:.4f}")
    print(f"  Target epsilon: {fairness_tolerance}")
    
    return model, metrics


def example_small_batch_inference():
    """
    Example 2: Small-Batch Inference (Primal-Dual Algorithm)
    
    Training uses large batches with hard constraints, but inference uses
    small batches with the online primal-dual algorithm (Algorithm 1).
    
    This simulates real-time/streaming deployment where:
    - Model is trained offline with large batches
    - Predictions are made in real-time with small batches or single samples
    
    Key settings:
    - Training batch_size = 2000 >= b_tau: Hard constraints
    - Inference batch_size = 64 < b_tau: Primal-dual algorithm
    
    Theorem 2.2 guarantees: aggregate fairness gap ≤ epsilon + O(T^{-1/4})
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: SMALL-BATCH INFERENCE (Online Primal-Dual Algorithm)")
    print("="*80)
    
    # Load data
    X, y, feature_names = load_and_preprocess_adult_data()
    if X is None:
        return None
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Configuration - LARGE training batches, SMALL inference batches
    batch_size_train = 2000  # Large batch for training (hard constraints)
    batch_size_eval = 64     # Small batch for inference (primal-dual)
    b_tau = 2000             # Threshold
    fairness_tolerance = 0.05
    
    # Create dataloaders with different batch sizes
    # Training: stratified large batches
    train_loader, val_loader_large = create_stratified_dataloaders(
        X_train, y_train, X_val, y_val,
        protected_attr_idx=0,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_train  # Large batches for validation during training
    )
    
    # Test: small batches (simulates streaming inference)
    _, _, test_loader_small = create_dataloaders(
        X_train, y_train,  # Dummy, won't be used
        X_val, y_val,      # Dummy, won't be used  
        X_test, y_test,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval  # Small batches for test
    )
    
    # Create model
    input_dim = X.shape[1]
    model = FairModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        protected_attr_idx=[0],  # Just gender for simplicity
        prediction_bounds=(-10.0, 10.0),
        fairness_tolerance=fairness_tolerance,
        b_tau=b_tau,
        eta_0=0.5,  # Initial dual step size
        fairness_criterion='mean_pred'
    )
    
    print(f"\nModel configuration:")
    print(f"  - Training:  batch_size={batch_size_train} >= b_tau={b_tau} -> HARD constraints")
    print(f"  - Inference: batch_size={batch_size_eval} < b_tau={b_tau} -> PRIMAL-DUAL algorithm")
    print(f"  - fairness_tolerance (epsilon)={fairness_tolerance}")
    print(f"  - η_0={model.eta_0} (dual step size)")
    print(f"  - Adaptive step size: η_t = η_0 / sqrt(t)")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = FairTrainer(
        model, criterion, optimizer, 
        scheduler=scheduler,
        early_stopping_patience=15
    )
    
    # Train with large batches
    print("\n--- Training (Large Batches - Hard Constraints) ---")
    history = trainer.fit(
        train_loader, val_loader_large, 
        epochs=30, 
        verbose=1, 
        log_interval=5
    )
    
    # Evaluate with small batches (primal-dual algorithm)
    print("\n--- Test Evaluation (Small Batches - Primal-Dual) ---")
    metrics = trainer.evaluate(test_loader_small, return_predictions=True)
    
    # Get aggregate fairness statistics
    aggregate_stats = model.get_aggregate_fairness_stats(test_loader_small, reset_before=True)
    
    print(f"\nResults (Small-Batch Primal-Dual Inference):")
    print(f"  Test Loss: {metrics['test_loss']:.4f}")
    print(f"  Aggregate Fairness Gap: {aggregate_stats['aggregate_gap']:.4f}")
    print(f"  Target epsilon: {fairness_tolerance}")
    print(f"  λ_max observed: {aggregate_stats['lambda_max']:.4f}")
    print(f"  Total inference samples: {aggregate_stats['total_samples']}")
    print(f"  Theoretical bound (Theorem 2.2): {aggregate_stats['theoretical_bound']:.4f}")
    
    return model, metrics

def example_mean_residual_fairness():
    """
    Example 4: Mean Residual Fairness for Regression
    
    Uses the mean residual fairness criterion: E[Y - Ŷ | A=a] ≈ 0 for all groups.
    This ensures the model doesn't systematically over/under-predict for any group.
    
    We use a synthetic regression dataset here to demonstrate.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: MEAN RESIDUAL FAIRNESS (Regression)")
    print("="*80)
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    # Protected attribute (binary)
    protected = np.random.binomial(1, 0.3, n_samples)  # 30% in group 1
    
    # Features (some correlated with protected attribute)
    X_features = np.random.randn(n_samples, n_features - 1)
    X_features[:, 0] += 0.5 * protected  # Add correlation with protected attribute
    
    # Combine: protected attribute as first column
    X = np.hstack([protected.reshape(-1, 1), X_features]).astype(np.float32)
    
    # True target with group bias (what we want to correct)
    y_true = 3 * X[:, 1] - 2 * X[:, 2] + X[:, 3] + np.random.randn(n_samples) * 0.5
    y_biased = y_true + 2.0 * protected  # Add bias: group 1 has higher targets
    y = y_biased.astype(np.float32)
    
    print(f"\nSynthetic regression data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Group 0: {(1-protected).sum()} samples, mean target={y[protected==0].mean():.2f}")
    print(f"  Group 1: {protected.sum()} samples, mean target={y[protected==1].mean():.2f}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Create dataloaders with large training batches
    batch_size_train = 2000
    batch_size_eval = 64  # Small batches for inference (primal-dual)
    
    train_loader, val_loader = create_stratified_dataloaders(
        X_train, y_train, X_val, y_val,
        protected_attr_idx=0,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_train
    )
    
    _, _, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval
    )
    
    # Create model with mean_residual fairness criterion
    model = FairModel(
        input_dim=n_features,
        hidden_dims=[64, 32],
        output_dim=1,
        protected_attr_idx=0,
        prediction_bounds=(-20.0, 20.0),  # Wider bounds for regression
        fairness_tolerance=0.1,  # Allow small mean residual
        b_tau=2000,
        fairness_criterion='mean_residual'  # KEY: Use mean residual fairness
    )
    
    print(f"\nModel configuration:")
    print(f"  - fairness_criterion='mean_residual'")
    print(f"  - Constraint: |E[Y - Ŷ | A=a]| ≤ epsilon for all groups")
    print(f"  - Training: batch_size={batch_size_train} (hard constraints)")
    print(f"  - Inference: batch_size={batch_size_eval} (primal-dual)")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = FairTrainer(
        model, criterion, optimizer, 
        scheduler=scheduler,
        early_stopping_patience=15
    )
    
    # Train
    print("\n--- Training ---")
    history = trainer.fit(
        train_loader, val_loader, 
        epochs=50, 
        verbose=1, 
        log_interval=10
    )
    
    # Evaluate
    metrics = trainer.evaluate(test_loader, return_predictions=True)
    
    print(f"\nResults (Mean Residual Fairness):")
    print(f"  Test Loss (MSE): {metrics['test_loss']:.4f}")
    print(f"  Mean Residual Group 0: {metrics.get('mean_residual_attr_0_group_0', 'N/A'):.4f}")
    print(f"  Mean Residual Group 1: {metrics.get('mean_residual_attr_0_group_1', 'N/A'):.4f}")
    print(f"  Fairness Gap: {metrics['fairness_gap']:.4f}")
    
    # Compare with utils evaluation
    print("\n--- Detailed Fairness Evaluation ---")
    residual_metrics = utils.compute_equalized_residuals(
        metrics['predictions'],
        metrics['targets'],
        metrics['protected'][0]
    )
    for key, val in residual_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FAIR NEURAL NETWORKS - EXAMPLES")
    print("="*80)
    print("\nKey Design:")
    print("  - Training: Always uses large batches with HARD per-batch constraints")
    print("  - Inference: Can use large batches (hard) or small batches (primal-dual)")
    
    # Run examples
    print("\nSelect an example to run:")
    print("  1: Large-batch inference (hard constraints throughout)")
    print("  2: Small-batch inference (primal-dual algorithm)")
    print("  3: Mean residual fairness (regression)")

    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        example_large_batch_inference()
    elif choice == '2':
        example_small_batch_inference()
    elif choice == '3':
        example_mean_residual_fairness()
    else:
        print("Invalid choice")
