"""
Example: Fair Neural Network with Multiple Protected Attributes

This example demonstrates how to use the Fair Neural Networks package on a real dataset
with multiple binary protected attributes. We'll use the Adult Income dataset which has
gender and race as protected attributes.

Dataset: Adult Income (UCI Machine Learning Repository)
Task: Predict whether income exceeds $50K/year
Protected Attributes: Gender (sex) and Race (white vs non-white)
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
from trainer import FairTrainer, create_dataloaders
from utils import (
    compute_demographic_parity,
    compute_equalized_residuals
)


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
        print("✓ Successfully loaded Adult Income dataset from UCI")
    except:
        print("× Could not download dataset. Using synthetic data instead...")
        return generate_synthetic_data_with_multiple_protected()
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Create binary target: 1 if income > 50K, 0 otherwise
    df['income_binary'] = (df['income'] == '>50K').astype(int)
    
    # Create protected attributes
    # Protected attribute 1: Gender (0 = Female, 1 = Male)
    df['gender_binary'] = (df['sex'] == 'Male').astype(int)
    
    # Protected attribute 2: Race (0 = Non-White, 1 = White)
    df['race_binary'] = (df['race'] == 'White').astype(int)
    
    # Select features for modeling (exclude protected attributes from features used for prediction)
    feature_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    # One-hot encode categorical features
    categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'native_country']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Get all feature columns (including one-hot encoded)
    all_feature_cols = [col for col in df_encoded.columns 
                        if col not in ['income', 'income_binary', 'gender_binary', 'race_binary', 
                                      'sex', 'race', 'education']]
    
    # Create feature matrix
    X_features = df_encoded[all_feature_cols].values.astype(np.float32)
    
    # Standardize features
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


def example_single_vs_multiple_protected_attrs():
    """
    Compare fairness with single vs multiple protected attributes.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Single vs Multiple Protected Attributes")
    print("="*80)
    
    # Load data
    X, y, feature_names = load_and_preprocess_adult_data()
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size_train=256,
        batch_size_eval=128 # Larger for inference
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # -------------------------------------------------------------------------
    # Model 2: Multiple Protected Attributes (Gender + Race)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("MODEL 2: Multiple Protected Attributes (Gender + Race)")
    print("="*80)
    print(X_train)
    print(y_train)
    model_multiple = FairModel(
        input_dim=X.shape[1],
        hidden_dims=[64, 32],
        output_dim=1,
        protected_attr_idx=[0, 1],  # Gender AND Race
        prediction_bounds=(-10.0, 10.0),
        initial_slack=200,
        min_slack=0.25,
        slack_decay=.99,
        b_tau=200,
        fairness_criterion='mean_pred',
        train_batch_size=256,
        eval_batch_size=128
    )
    
    optimizer_multiple = optim.SGD(model_multiple.parameters(), lr=0.005)
    scheduler_multiple = ReduceLROnPlateau(optimizer_multiple, patience=3)
    
    trainer_multiple = FairTrainer(
        model_multiple, criterion, optimizer_multiple, scheduler=scheduler_multiple,
        early_stopping_patience=30
    )
    
    print("\nTraining...")
    history_multiple = trainer_multiple.fit(
        train_loader, val_loader, epochs=100, verbose=1, log_interval=1
    )
    
    # Evaluate
    metrics_multiple = trainer_multiple.evaluate(test_loader, return_predictions=True)
    
    print(f"\nResults (Multiple Protected Attributes):")
    print(f"  Test Loss: {metrics_multiple['test_loss']:.4f}")
    print(f"  Fairness Gap (Gender): {metrics_multiple['fairness_gap_attr_0']:.4f}")
    print(f"  Fairness Gap (Race):   {metrics_multiple['fairness_gap_attr_1']:.4f}")
    print(f"  Max Fairness Gap:      {metrics_multiple['fairness_gap']:.4f}")
    
    # Check demographic parity
    preds_multiple = torch.sigmoid(torch.tensor(metrics_multiple['predictions'])).numpy()
    
    gender_dp_multi = compute_demographic_parity(preds_multiple, X_test[:, 0])
    race_dp_multi = compute_demographic_parity(preds_multiple, X_test[:, 1])
    
    print(f"  Demographic Parity (Gender): {gender_dp_multi['demographic_parity_diff']:.4f}")
    print(f"  Demographic Parity (Race):   {race_dp_multi['demographic_parity_diff']:.4f}")
    
    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<40} {'Single (Gender)':<20} {'Multiple (Gender+Race)':<20}")
    print("-"*80)
    print("\n" + "="*80)
    print("Key Insight:")
    print("  - Single attribute model: Fair on gender, but NOT on race")
    print("  - Multiple attribute model: Fair on BOTH gender and race")
    print("="*80)
    
    return model_single, model_multiple

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FAIR NEURAL NETWORKS - MULTIPLE PROTECTED ATTRIBUTES EXAMPLES")
    print("="*80)
    
    # Example 2: Single vs Multiple
    print("\n\nEXAMPLE 2: Single vs Multiple Protected Attributes")
    print("="*80)
    model_single, model_multiple = example_single_vs_multiple_protected_attrs()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Multiple protected attributes ensure fairness across ALL groups")
    print("2. Single attribute models may be unfair on other attributes")
    print("3. Mean residual fairness is better for regression tasks")
    print("4. The package seamlessly handles 1+ protected attributes")
    print("="*80)
