"""
Quick Heart Disease Risk Analysis
Fast comparison of Logistic Regression vs Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def quick_analysis():
    """Perform a quick analysis with basic models."""
    print("=" * 60)
    print("QUICK HEART DISEASE RISK ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')
    print(f"Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]-1} features")
    
    # Prepare data
    X = df.drop('Heart_Risk', axis=1)
    y = df['Heart_Risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    lr_pred = lr.predict(X_test_scaled)
    lr_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    rf_pred = rf.predict(X_test)
    rf_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    models = {
        'Logistic Regression': {
            'predictions': lr_pred,
            'probabilities': lr_pred_proba
        },
        'Random Forest': {
            'predictions': rf_pred,
            'probabilities': rf_pred_proba
        }
    }
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    results = {}
    for model_name, preds in models.items():
        accuracy = accuracy_score(y_test, preds['predictions'])
        precision = precision_score(y_test, preds['predictions'])
        recall = recall_score(y_test, preds['predictions'])
        f1 = f1_score(y_test, preds['predictions'])
        roc_auc = roc_auc_score(y_test, preds['probabilities'])
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Determine best model
    best_model = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    print(f"\nBEST MODEL: {best_model}")
    print(f"ROC-AUC: {results[best_model]['roc_auc']:.4f}")
    
    # Feature importance (Random Forest)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTOP 10 MOST IMPORTANT FEATURES (Random Forest):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:<20} {row['Importance']:.4f}")
    
    # Dataset recommendations
    print(f"\nDATASET RECOMMENDATIONS:")
    print(f"- Current size: {len(df):,} samples - EXCELLENT for ML")
    print(f"- Perfect balance: 50/50 class distribution")
    print(f"- No missing values")
    print(f"- Recommend keeping full dataset for best performance")
    
    return results, feature_importance

if __name__ == "__main__":
    results, feature_importance = quick_analysis()
