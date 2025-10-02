"""
Model Training Script for Heart Disease Risk Prediction
This script trains and saves both Logistic Regression and Random Forest models for deployment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_prepare_data():
    """Load and prepare the dataset for training."""
    print("Loading dataset...")
    df = pd.read_csv('../data/heart_disease_risk_dataset_earlymed.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['Heart_Risk'].value_counts()}")
    
    # Prepare features and target
    X = df.drop('Heart_Risk', axis=1)
    y = df['Heart_Risk']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, X.columns

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with hyperparameter tuning."""
    print("Training Logistic Regression...")
    
    # Define parameter grid (optimized for speed)
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    # Create and train model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(lr_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning."""
    print("Training Random Forest...")
    
    # Define parameter grid (optimized for speed)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create and train model
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def save_models_and_metadata(models_data, feature_names):
    """Save the trained models and metadata."""
    print("Saving models and metadata...")
    
    # Create combined model data dictionary
    combined_data = {
        'models': models_data,
        'feature_names': feature_names.tolist(),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': '70,000 samples',
        'features': len(feature_names)
    }
    
    # Save to pickle file
    with open('heart_disease_models.pkl', 'wb') as f:
        pickle.dump(combined_data, f)
    
    print("Models saved as 'heart_disease_models.pkl'")
    
    # Save feature names separately for easy access
    with open('feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print("Feature names saved as 'feature_names.txt'")

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("HEART DISEASE RISK PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Scale features for Logistic Regression
    print("Scaling features for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test)
    
    # Train Random Forest (no scaling needed)
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    
    # Prepare models data
    models_data = {
        'logistic_regression': {
            'model': lr_model,
            'scaler': scaler,
            'performance_metrics': lr_metrics,
            'model_info': {
                'algorithm': 'Logistic Regression',
                'best_params': lr_model.get_params(),
                'needs_scaling': True
            }
        },
        'random_forest': {
            'model': rf_model,
            'scaler': None,  # Random Forest doesn't need scaling
            'performance_metrics': rf_metrics,
            'model_info': {
                'algorithm': 'Random Forest',
                'best_params': rf_model.get_params(),
                'needs_scaling': False
            }
        }
    }
    
    # Save models and metadata
    save_models_and_metadata(models_data, feature_names)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Files created:")
    print("- heart_disease_models.pkl (both trained models)")
    print("- feature_names.txt (feature list)")
    print("\nModel Performance Summary:")
    print(f"Logistic Regression - Accuracy: {lr_metrics['accuracy']:.4f}, ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print("\nReady for deployment!")

if __name__ == "__main__":
    main()
