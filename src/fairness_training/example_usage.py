"""
Example: Fairness Layers with Differentiable Fairness Constraints

This module demonstrates how to use the Fairness Layers package with the
primal-dual algorithm for enforcing fairness constraints during inference.

Key Insight: Training always uses large batches with hard per-batch constraints.
             Inference can use either:
             - Large batches: Hard per-batch constraints
             - Small batches: Online primal-dual algorithm with 
                              provable aggregate fairness

Examples included:
1. Large-batch inference: Hard constraints for both training and inference
2. Small-batch inference: Hard constraints for training, primal-dual for inference
3. Mean residual fairness for regression tasks
4. Custom fairness metric

Dataset: Adult Income (UCI Machine Learning Repository)
Task: Predict whether income exceeds $50K/year
Protected Attributes: Gender (sex) and Race (white vs non-white)
"""

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from fair_model import FairModel
from fair_trainer import FairTrainer
from utils import create_dataloaders, create_stratified_dataloaders


def load_and_preprocess_adult_data():
    """
    Load and preprocess the Adult Income UCI dataset.
    
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
        print(f"Could not download dataset: {e}")
        return None, None, None
    

    df = df.dropna()
    df['income_binary'] = (df['income'] == '>50K').astype(int)
    
    # Create protected attributes, must be binary
    df['gender_binary'] = (df['sex'] == 'Male').astype(int)
    df['race_binary'] = (df['race'] == 'White').astype(int)
    
    #one-hot encode categorical features
    categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'native_country']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    all_feature_cols = [col for col in df_encoded.columns 
                        if col not in ['income', 'income_binary', 'gender_binary', 'race_binary', 
                                      'sex', 'race', 'education']]
    
    # Create feature matrix w/ protected attributes as first two columns
    X_features = df_encoded[all_feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)
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
    Example: Large-Batch Inference. Enforce hard constraints per mini-batch.
    
    Both training and inference use large batches with hard per-batch fairness
    constraints
    
    Key settings:
    - Training batch_size = 2000 >= b_tau = 2000: Hard constraints
    - Inference batch_size = 2000 >= b_tau = 2000: Hard constraints
    - Uses create_stratified_dataloaders() to maintain constant group ratios
    """
    print("\n" + "="*80)
    print("EXAMPLE: LARGE-BATCH INFERENCE")
    print("="*80)
    
    X, y, feature_names = load_and_preprocess_adult_data()
    if X is None:
        return None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    batch_size_train = 2000  
    batch_size_eval = 2000   
    b_tau = 2000             # Threshold for hard constraints
    fairness_tolerance = 0.05
    
    # Create stratified dataloaders (maintains constant group ratios)
    train_loader, val_loader, test_loader = create_stratified_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        protected_attr_idx=[0,1], # 2 protected attributes
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval
    )
    
    input_dim = X.shape[1]

    #Load instance of FairModel and FairTrainer

    model = FairModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        protected_attr_idx=[0, 1],
        prediction_bounds=(-10.0, 10.0),
        fairness_tolerance=fairness_tolerance,
        b_tau=b_tau,
        fairness_metric='mean_pred'
    )
    
    print(f"\nModel configuration:")
    print(f"  - Training:  batch_size={batch_size_train} >= b_tau={b_tau} -> HARD constraints")
    print(f"  - Inference: batch_size={batch_size_eval} >= b_tau={b_tau} -> HARD constraints")
    print(f"  - fairness_tolerance (epsilon)={fairness_tolerance}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = FairTrainer(
        model, criterion, optimizer, 
        scheduler=scheduler,
        early_stopping_patience=15
    )
    
    print("\n--- Training ---")
    history = trainer.fit(
        train_loader, val_loader, 
        epochs=50, 
        verbose=1, 
        log_interval=5
    )
    
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
    Example: Small-Batch Inference (online Primal-Dual Inference Algorithm)
    
    Training uses large batches with hard constraints, but inference uses
    small batches with the online primal-dual algorithm
    
    Simulates real-time/streaming deployment where:
    - Model is trained offline with large batches
    - Predictions are made in real-time with small batches or single samples
    
    Key settings:
    - Training batch_size = 2000 >= b_tau: Hard constraints
    - Inference batch_size = 64 < b_tau: Primal-dual algorithm
    
    """
    print("\n" + "="*80)
    print("EXAMPLE: SMALL-BATCH INFERENCE (Online Primal-Dual Algorithm)")
    print("="*80)
    
    X, y, feature_names = load_and_preprocess_adult_data()
    if X is None:
        return None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    batch_size_train = 2000  
    batch_size_eval = 64
    b_tau = 2000             # Threshold for hard constraints
    fairness_tolerance = 0.05
    
    # Create dataloaders with different batch sizes
    train_loader, val_loader_large = create_stratified_dataloaders(
        X_train, y_train, X_val, y_val,
        protected_attr_idx=[0,1],
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval  # Large batches for validation during training
    )
    
    _, _, test_loader_small = create_dataloaders(
        X_train, y_train,  # Dummy, won't be used
        X_val, y_val,      # Dummy, won't be used  
        X_test, y_test,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval  # Small batches for test
    )
    
    # Create instance of FairModel and FairTrainer
    input_dim = X.shape[1]
    model = FairModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        protected_attr_idx=[0],  # Single protected attr
        prediction_bounds=(-10.0, 10.0),
        fairness_tolerance=fairness_tolerance,
        b_tau=b_tau,
        eta_0=0.5,  # Initial dual step size
        fairness_metric='mean_pred'
    )
    
    print(f"\nModel configuration:")
    print(f"  - Training:  batch_size={batch_size_train} >= b_tau={b_tau} -> HARD constraints")
    print(f"  - Inference: batch_size={batch_size_eval} < b_tau={b_tau} -> PRIMAL-DUAL algorithm")
    print(f"  - fairness_tolerance (epsilon)={fairness_tolerance}")
    print(f"  - eta_0={model.eta_0} (dual step size)")
    print(f"  - Adaptive step size: eta_t = eta_0 / sqrt(t)")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = FairTrainer(
        model, criterion, optimizer, 
        scheduler=scheduler,
        early_stopping_patience=15
    )
    print("\n--- Training (Large Batches - Hard Constraints) ---")
    history = trainer.fit(
        train_loader, val_loader_large, 
        epochs=30, 
        verbose=1, 
        log_interval=5
    )
        
    print("\n--- Test Evaluation (Small Batches - Primal-Dual) ---")
    metrics = trainer.evaluate(test_loader_small, return_predictions=True)
    aggregate_stats = model.get_aggregate_fairness_stats(test_loader_small, reset_before=True)
    
    print(f"\nResults (Small-Batch Primal-Dual Inference):")
    print(f"  Test Loss: {metrics['test_loss']:.4f}")
    print(f"  Aggregate Fairness Gap: {aggregate_stats['aggregate_gap']:.4f}")
    print(f"  Target epsilon: {fairness_tolerance}")
    print(f"  lambda_max observed: {aggregate_stats['lambda_max']:.4f}")
    print(f"  Total inference samples: {aggregate_stats['total_samples']}")

    return model, metrics

def example_mean_residual_fairness():
    """
    Example: Mean Residual Fairness for Regression
    
    Uses the mean residual fairness criterion: E[Y - Y_hat | A=a] <= tolerance for all groups.
    This ensures the model doesn't systematically over/under-predict for any group.
    
    We use a synthetic regression dataset here to demonstrate.
    """
    print("\n" + "="*80)
    print("EXAMPLE: MEAN RESIDUAL FAIRNESS (Regression)")
    print("="*80)
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    protected = np.random.binomial(1, 0.3, n_samples)  # 30% in group 1
    X_features = np.random.randn(n_samples, n_features - 1)
    X_features[:, 0] += 0.5 * protected  # Add correlation with protected attribute
    X = np.hstack([protected.reshape(-1, 1), X_features]).astype(np.float32)
    y_true = 3 * X[:, 1] - 2 * X[:, 2] + X[:, 3] + np.random.randn(n_samples) * 0.5
    y_biased = y_true + 2.0 * protected  # Add some bias: group 1 has higher targets
    y = y_biased.astype(np.float32)
    
    print(f"\nSynthetic regression data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Group 0: {(1-protected).sum()} samples, mean target={y[protected==0].mean():.2f}")
    print(f"  Group 1: {protected.sum()} samples, mean target={y[protected==1].mean():.2f}")
    

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
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
    
    # Create FairModel and FairTrainer
    model = FairModel(
        input_dim=n_features,
        hidden_dims=[64, 32],
        output_dim=1,
        protected_attr_idx=0,
        prediction_bounds=(-20.0, 20.0),
        fairness_tolerance=0.1,
        b_tau=2000,
        fairness_metric='mean_residual'
    )
    
    print(f"\nModel configuration:")
    print(f"  - fairness_metric='mean_residual'")
    print(f"  - Constraint: |E[Y - Y_hat | A=a]| <= epsilon for all groups")
    print(f"  - Training: batch_size={batch_size_train} (hard constraints)")
    print(f"  - Inference: batch_size={batch_size_eval} (primal-dual)")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = FairTrainer(
        model, criterion, optimizer, 
        scheduler=scheduler,
        early_stopping_patience=15
    )
    
    print("\n--- Training ---")
    history = trainer.fit(
        train_loader, val_loader, 
        epochs=50, 
        verbose=1, 
        log_interval=10
    )
    
    metrics = trainer.evaluate(test_loader, return_predictions=True)
    
    print(f"\nResults (Mean Residual Fairness):")
    print(f"  Test Loss (MSE): {metrics['test_loss']:.4f}")
    print(f"  Fairness Gap: {metrics.get('fairness_gap_attr_0', 'N/A'):.4f}")
    
    return model, metrics

def example_custom_fairness_metric():
    """
    Example: Use custom fairness metric with FairModel. Uses Equalized Odds metric
    inherited from FairnessMetric base class.

    """
    print("\n" + "="*80)
    print("EXAMPLE: CUSTOM FAIRNESS METRIC")
    print("="*80)
    
    X, y, feature_names = load_and_preprocess_adult_data()
    if X is None:
        return None
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y #stratify also on y since metric depends on y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    input_dim = X.shape[1]
    batch_size_train = 6000
    batch_size_eval = 6000
    fairness_tolerance = 0.05
    
    train_loader, val_loader, test_loader = create_stratified_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        protected_attr_idx=0,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval
    )

    import fairness_metrics
    
    # Create custom metric
    custom_metric = fairness_metrics.EqualizedOdds( num_protected_attrs=1 )
    
    print(f"Custom Metric: {type(custom_metric).__name__}")
    print(f"  - requires_targets: {custom_metric.requires_targets}")
    
    # Create FairModel and FairTrainer
    model = FairModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        protected_attr_idx=0,
        prediction_bounds=(-10.0, 10.0),
        fairness_tolerance=fairness_tolerance,
        b_tau=1000,
        fairness_metric=custom_metric
    )
    
    print(f"\nFairModel created with:")
    print(f"  - Custom metric: {type(model.fairness_metric).__name__}")
    print(f"  - requires_targets: {model.requires_targets}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = FairTrainer(
        model, criterion, optimizer,
        scheduler=scheduler,
        early_stopping_patience=15
    )
    
    history = trainer.fit(
        train_loader, val_loader,
        epochs=20,
        verbose=1,
        log_interval=5
    )
    
    print("\n--- Test Evaluation ---")
    metrics = trainer.evaluate(test_loader, return_predictions=True)
    print(f"\nOverall Results:")
    print(f"  Test Loss: {metrics['test_loss']:.4f}")
    print(f"  Max Fairness Gap: {metrics['fairness_gap_attr_0']:.4f}")
    print(f"  Target tolerance: {fairness_tolerance}")
    
    return model, metrics


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FAIR NEURAL NETWORKS - EXAMPLES")
    print("="*80)

    # Run examples
    print("\nSelect an example to run:")
    print("  1: Large-batch inference (hard constraints throughout)")
    print("  2: Small-batch inference (primal-dual algorithm)")
    print("  3: Mean residual fairness (regression)")
    print("  4: Custom fairness metric")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        example_large_batch_inference()
    elif choice == '2':
        example_small_batch_inference()
    elif choice == '3':
        example_mean_residual_fairness()
    elif choice == '4':
        example_custom_fairness_metric()
    else:
        print("Invalid choice.")
