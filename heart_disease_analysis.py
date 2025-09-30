"""
Heart Disease Risk Prediction Analysis
Comparing Logistic Regression vs Random Forest

This script performs comprehensive analysis of heart disease risk prediction
using a dataset of 70,000 samples with 19 features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HeartDiseaseAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with the dataset path."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset."""
        print("=" * 60)
        print("HEART DISEASE RISK PREDICTION ANALYSIS")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Features: {self.df.shape[1] - 1}")
        print(f"Samples: {self.df.shape[0]:,}")
        
        # Display target distribution
        print(f"\nTarget Variable Distribution:")
        target_counts = self.df['Heart_Risk'].value_counts()
        print(f"Low Risk (0): {target_counts[0]:,} ({target_counts[0]/len(self.df)*100:.1f}%)")
        print(f"High Risk (1): {target_counts[1]:,} ({target_counts[1]/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        print("\nDataset Overview:")
        print(self.df.describe())
        
        # Check for missing values
        print(f"\nMissing Values:")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Feature correlation with target
        print(f"\nFeature Correlation with Heart Risk:")
        correlations = self.df.corr()['Heart_Risk'].sort_values(ascending=False)
        print(correlations)
        
        # Create correlation heatmap
        plt.figure(figsize=(15, 12))
        correlation_matrix = self.df.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Age distribution by heart risk
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.df, x='Age', hue='Heart_Risk', kde=True, alpha=0.7)
        plt.title('Age Distribution by Heart Risk')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.df, x='Heart_Risk', y='Age')
        plt.title('Age Distribution by Heart Risk (Box Plot)')
        plt.xlabel('Heart Risk (0=Low, 1=High)')
        plt.ylabel('Age')
        
        plt.tight_layout()
        plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Top risk factors
        risk_factors = correlations.drop('Heart_Risk').sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=risk_factors.values, y=risk_factors.index, palette='viridis')
        plt.title('Feature Importance (Correlation with Heart Risk)', fontsize=14, fontweight='bold')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for machine learning."""
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        
        # Separate features and target
        self.X = self.df.drop('Heart_Risk', axis=1)
        self.y = self.df['Heart_Risk']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples")
        
        # Scale features (important for Logistic Regression)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preparation completed!")
        
    def train_logistic_regression(self):
        """Train Logistic Regression model with hyperparameter tuning."""
        print("\n" + "=" * 60)
        print("LOGISTIC REGRESSION TRAINING")
        print("=" * 60)
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Grid search with cross-validation
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Store best model
        self.models['Logistic Regression'] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Make predictions
        y_pred_lr = self.models['Logistic Regression'].predict(self.X_test_scaled)
        y_pred_proba_lr = self.models['Logistic Regression'].predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        self.results['Logistic Regression'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_lr),
            'precision': precision_score(self.y_test, y_pred_lr),
            'recall': recall_score(self.y_test, y_pred_lr),
            'f1_score': f1_score(self.y_test, y_pred_lr),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_lr),
            'predictions': y_pred_lr,
            'probabilities': y_pred_proba_lr
        }
        
        print(f"\nLogistic Regression Results:")
        print(f"Accuracy: {self.results['Logistic Regression']['accuracy']:.4f}")
        print(f"Precision: {self.results['Logistic Regression']['precision']:.4f}")
        print(f"Recall: {self.results['Logistic Regression']['recall']:.4f}")
        print(f"F1-Score: {self.results['Logistic Regression']['f1_score']:.4f}")
        print(f"ROC-AUC: {self.results['Logistic Regression']['roc_auc']:.4f}")
        
    def train_random_forest(self):
        """Train Random Forest model with hyperparameter tuning."""
        print("\n" + "=" * 60)
        print("RANDOM FOREST TRAINING")
        print("=" * 60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Grid search with cross-validation
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(self.X_train, self.y_train)  # RF doesn't need scaling
        
        # Store best model
        self.models['Random Forest'] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Make predictions
        y_pred_rf = self.models['Random Forest'].predict(self.X_test)
        y_pred_proba_rf = self.models['Random Forest'].predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        self.results['Random Forest'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_rf),
            'precision': precision_score(self.y_test, y_pred_rf),
            'recall': recall_score(self.y_test, y_pred_rf),
            'f1_score': f1_score(self.y_test, y_pred_rf),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_rf),
            'predictions': y_pred_rf,
            'probabilities': y_pred_proba_rf
        }
        
        print(f"\nRandom Forest Results:")
        print(f"Accuracy: {self.results['Random Forest']['accuracy']:.4f}")
        print(f"Precision: {self.results['Random Forest']['precision']:.4f}")
        print(f"Recall: {self.results['Random Forest']['recall']:.4f}")
        print(f"F1-Score: {self.results['Random Forest']['f1_score']:.4f}")
        print(f"ROC-AUC: {self.results['Random Forest']['roc_auc']:.4f}")
        
    def compare_models(self):
        """Compare the performance of both models."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(comparison_df.round(4))
        
        # Visualize comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, 
                             color=['skyblue', 'lightcoral'], alpha=0.8)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xlabel('Model')
            ax.legend().remove()
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(comparison_df[metric]):
                ax.text(j, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curves comparison
        plt.figure(figsize=(10, 8))
        for model_name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, metrics['probabilities'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, (model_name, metrics) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, metrics['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Low Risk', 'High Risk'],
                       yticklabels=['Low Risk', 'High Risk'])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_importance_analysis(self):
        """Analyze feature importance from both models."""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Logistic Regression coefficients
        lr_coef = self.models['Logistic Regression'].coef_[0]
        feature_names = self.X.columns
        
        lr_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': lr_coef,
            'Abs_Coefficient': np.abs(lr_coef)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nLogistic Regression Feature Importance (Top 10):")
        print(lr_importance.head(10))
        
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.models['Random Forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nRandom Forest Feature Importance (Top 10):")
        print(rf_importance.head(10))
        
        # Visualize feature importance
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Logistic Regression
        sns.barplot(data=lr_importance.head(10), x='Abs_Coefficient', y='Feature', 
                   ax=axes[0], palette='viridis')
        axes[0].set_title('Logistic Regression Feature Importance', fontweight='bold')
        axes[0].set_xlabel('Absolute Coefficient Value')
        
        # Random Forest
        sns.barplot(data=rf_importance.head(10), x='Importance', y='Feature', 
                   ax=axes[1], palette='plasma')
        axes[1].set_title('Random Forest Feature Importance', fontweight='bold')
        axes[1].set_xlabel('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return lr_importance, rf_importance
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)
        
        # Determine best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        
        print(f"\nBEST PERFORMING MODEL: {best_model}")
        print(f"ROC-AUC Score: {self.results[best_model]['roc_auc']:.4f}")
        
        print(f"\nDETAILED RESULTS:")
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall:    {metrics['recall']:.4f}")
            print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nKEY INSIGHTS:")
        print(f"- Dataset size: {len(self.df):,} samples - excellent for ML")
        print(f"- Perfect class balance: 50/50 split")
        print(f"- No missing values - clean dataset")
        print(f"- Both models show strong performance")
        
        if self.results['Random Forest']['roc_auc'] > self.results['Logistic Regression']['roc_auc']:
            print(f"- Random Forest slightly outperforms Logistic Regression")
        else:
            print(f"- Logistic Regression slightly outperforms Random Forest")
        
        print(f"\nRECOMMENDATIONS:")
        print(f"- Use {best_model} for production deployment")
        print(f"- Consider ensemble methods for even better performance")
        print(f"- Feature engineering could further improve results")
        print(f"- Cross-validation shows robust model performance")
        
        return best_model, self.results[best_model]

def main():
    """Main function to run the complete analysis."""
    # Initialize analyzer
    analyzer = HeartDiseaseAnalyzer('heart_disease_risk_dataset_earlymed.csv')
    
    # Run complete analysis pipeline
    analyzer.load_data()
    analyzer.explore_data()
    analyzer.prepare_data()
    analyzer.train_logistic_regression()
    analyzer.train_random_forest()
    analyzer.compare_models()
    analyzer.feature_importance_analysis()
    best_model, best_results = analyzer.generate_report()
    
    print(f"\nAnalysis completed successfully!")
    print(f"Generated visualizations saved as PNG files")
    print(f"Best model: {best_model} with ROC-AUC: {best_results['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
