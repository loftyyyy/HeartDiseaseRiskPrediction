"""
Analysis of Duplicate Impact on Model Performance
Compare results with and without duplicates
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_duplicate_impact():
    """Compare model performance with and without duplicates."""
    print("=" * 70)
    print("DUPLICATE IMPACT ANALYSIS")
    print("=" * 70)
    
    # Load original dataset
    df_original = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')
    print(f"Original dataset: {len(df_original):,} samples")
    
    # Load clean dataset (without duplicates)
    df_clean = pd.read_csv('heart_disease_risk_dataset_clean.csv')
    print(f"Clean dataset: {len(df_clean):,} samples")
    print(f"Removed duplicates: {len(df_original) - len(df_clean):,} samples")
    
    # Prepare data for both datasets
    datasets = {
        'Original (with duplicates)': df_original,
        'Clean (no duplicates)': df_clean
    }
    
    results_comparison = {}
    
    for dataset_name, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*50}")
        
        # Prepare data
        X = df.drop('Heart_Risk', axis=1)
        y = df['Heart_Risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {}
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        lr_metrics = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1_score': f1_score(y_test, lr_pred),
            'roc_auc': roc_auc_score(y_test, lr_pred_proba)
        }
        
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1_score': f1_score(y_test, rf_pred),
            'roc_auc': roc_auc_score(y_test, rf_pred_proba)
        }
        
        results_comparison[dataset_name] = {
            'Logistic Regression': lr_metrics,
            'Random Forest': rf_metrics
        }
        
        print(f"\n{dataset_name} Results:")
        print(f"Logistic Regression - ROC-AUC: {lr_metrics['roc_auc']:.4f}")
        print(f"Random Forest - ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    
    # Compare results
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    comparison_data = []
    for dataset_name, models in results_comparison.items():
        for model_name, metrics in models.items():
            comparison_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nDetailed Comparison:")
    print(comparison_df.round(4))
    
    # Calculate differences
    print(f"\n{'='*70}")
    print("PERFORMANCE DIFFERENCES (Original - Clean)")
    print(f"{'='*70}")
    
    for model in ['Logistic Regression', 'Random Forest']:
        orig_lr = results_comparison['Original (with duplicates)'][model]
        clean_lr = results_comparison['Clean (no duplicates)'][model]
        
        print(f"\n{model}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            diff = orig_lr[metric] - clean_lr[metric]
            print(f"  {metric.capitalize()}: {diff:+.4f}")
    
    # Statistical significance test
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    
    # Check if differences are meaningful
    lr_orig = results_comparison['Original (with duplicates)']['Logistic Regression']['roc_auc']
    lr_clean = results_comparison['Clean (no duplicates)']['Logistic Regression']['roc_auc']
    rf_orig = results_comparison['Original (with duplicates)']['Random Forest']['roc_auc']
    rf_clean = results_comparison['Clean (no duplicates)']['Random Forest']['roc_auc']
    
    print(f"Logistic Regression ROC-AUC difference: {lr_orig - lr_clean:.4f}")
    print(f"Random Forest ROC-AUC difference: {rf_orig - rf_clean:.4f}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    if abs(lr_orig - lr_clean) < 0.01 and abs(rf_orig - rf_clean) < 0.01:
        print("Duplicates have MINIMAL impact on performance")
        print("Keep original dataset for better statistical power")
        print("Results are robust and reliable")
    else:
        print("Duplicates may be affecting performance")
        print("Consider using clean dataset for more realistic results")
        print("Further investigation recommended")
    
    return results_comparison

def analyze_duplicate_patterns():
    """Analyze the nature of duplicate patterns."""
    print(f"\n{'='*70}")
    print("DUPLICATE PATTERN ANALYSIS")
    print(f"{'='*70}")
    
    df = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')
    
    # Find duplicates
    duplicate_mask = df.duplicated(keep=False)
    duplicates = df[duplicate_mask]
    
    print(f"Total duplicates: {len(duplicates):,}")
    print(f"Percentage of dataset: {len(duplicates)/len(df)*100:.2f}%")
    
    # Analyze duplicate characteristics
    print(f"\nDuplicate characteristics:")
    print(f"Age range: {duplicates['Age'].min():.0f} - {duplicates['Age'].max():.0f}")
    print(f"Gender distribution: {duplicates['Gender'].value_counts().to_dict()}")
    print(f"Heart risk distribution: {duplicates['Heart_Risk'].value_counts().to_dict()}")
    
    # Most common duplicate patterns
    print(f"\nTop 5 most common duplicate patterns:")
    duplicate_groups = df.groupby(list(df.columns)).size().reset_index(name='count')
    duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1].sort_values('count', ascending=False)
    
    for i, (_, row) in enumerate(duplicate_groups.head(5).iterrows(), 1):
        print(f"{i}. Count: {row['count']}, Age: {row['Age']}, Gender: {row['Gender']}, Heart_Risk: {row['Heart_Risk']}")
        print(f"   Symptoms: Chest_Pain={row['Chest_Pain']}, Shortness_of_Breath={row['Shortness_of_Breath']}")
    
    return duplicate_groups

if __name__ == "__main__":
    # Run duplicate impact analysis
    results = analyze_duplicate_impact()
    
    # Analyze duplicate patterns
    duplicate_patterns = analyze_duplicate_patterns()
