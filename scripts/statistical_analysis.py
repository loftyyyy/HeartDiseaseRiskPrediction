"""
Statistical Analysis for Heart Disease Risk Dataset
Comprehensive statistical summary including descriptive statistics, missing values, and data quality metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    def __init__(self, data_path):
        """Initialize the statistical analyzer with the dataset path."""
        self.data_path = data_path
        self.df = None
        self.stats_summary = None
        
    def load_data(self):
        """Load the dataset."""
        print("=" * 80)
        print("STATISTICAL ANALYSIS FOR HEART DISEASE RISK DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def comprehensive_statistical_summary(self):
        """Generate comprehensive statistical summary for all columns."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE STATISTICAL SUMMARY")
        print("=" * 80)
        
        # Initialize results dictionary
        stats_results = {}
        
        for column in self.df.columns:
            print(f"\n{'='*60}")
            print(f"ANALYSIS FOR COLUMN: {column}")
            print(f"{'='*60}")
            
            # Basic information
            print(f"Data Type: {self.df[column].dtype}")
            print(f"Total Values: {len(self.df[column]):,}")
            
            # Missing values analysis
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df[column])) * 100
            print(f"Missing Values: {missing_count:,} ({missing_percentage:.4f}%)")
            
            # Unique values
            unique_count = self.df[column].nunique()
            print(f"Unique Values: {unique_count:,}")
            
            # For numerical columns
            if self.df[column].dtype in ['int64', 'float64']:
                # Descriptive statistics
                mean_val = self.df[column].mean()
                median_val = self.df[column].median()
                std_val = self.df[column].std()
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                
                print(f"\nDESCRIPTIVE STATISTICS:")
                print(f"Mean:     {mean_val:.4f}")
                print(f"Median:   {median_val:.4f}")
                print(f"Std Dev:  {std_val:.4f}")
                print(f"Min:      {min_val:.4f}")
                print(f"Max:      {max_val:.4f}")
                
                # Additional statistics
                q25 = self.df[column].quantile(0.25)
                q75 = self.df[column].quantile(0.75)
                iqr = q75 - q25
                skewness = self.df[column].skew()
                kurtosis = self.df[column].kurtosis()
                
                print(f"\nADDITIONAL STATISTICS:")
                print(f"Q1 (25%): {q25:.4f}")
                print(f"Q3 (75%): {q75:.4f}")
                print(f"IQR:      {iqr:.4f}")
                print(f"Skewness: {skewness:.4f}")
                print(f"Kurtosis: {kurtosis:.4f}")
                
                # Value distribution
                print(f"\nVALUE DISTRIBUTION:")
                value_counts = self.df[column].value_counts().head(10)
                for value, count in value_counts.items():
                    percentage = (count / len(self.df[column])) * 100
                    print(f"  {value}: {count:,} ({percentage:.2f}%)")
                
                # Store results
                stats_results[column] = {
                    'data_type': str(self.df[column].dtype),
                    'total_values': len(self.df[column]),
                    'missing_values': missing_count,
                    'missing_percentage': missing_percentage,
                    'unique_values': unique_count,
                    'mean': mean_val,
                    'median': median_val,
                    'std_dev': std_val,
                    'min': min_val,
                    'max': max_val,
                    'q1': q25,
                    'q3': q75,
                    'iqr': iqr,
                    'skewness': skewness,
                    'kurtosis': kurtosis
                }
                
            else:
                # For non-numerical columns
                print(f"\nCATEGORICAL STATISTICS:")
                value_counts = self.df[column].value_counts()
                print(f"Most frequent value: {value_counts.index[0]} ({value_counts.iloc[0]:,} occurrences)")
                print(f"Least frequent value: {value_counts.index[-1]} ({value_counts.iloc[-1]:,} occurrences)")
                
                # Store results
                stats_results[column] = {
                    'data_type': str(self.df[column].dtype),
                    'total_values': len(self.df[column]),
                    'missing_values': missing_count,
                    'missing_percentage': missing_percentage,
                    'unique_values': unique_count,
                    'most_frequent': value_counts.index[0],
                    'most_frequent_count': value_counts.iloc[0],
                    'least_frequent': value_counts.index[-1],
                    'least_frequent_count': value_counts.iloc[-1]
                }
        
        self.stats_summary = stats_results
        return stats_results
    
    def create_summary_table(self):
        """Create a comprehensive summary table."""
        print("\n" + "=" * 80)
        print("SUMMARY TABLE - ALL COLUMNS")
        print("=" * 80)
        
        # Create summary DataFrame
        summary_data = []
        
        for column, stats in self.stats_summary.items():
            if 'mean' in stats:  # Numerical column
                summary_data.append({
                    'Column': column,
                    'Data_Type': stats['data_type'],
                    'Missing_Values': stats['missing_values'],
                    'Missing_%': f"{stats['missing_percentage']:.4f}%",
                    'Unique_Values': stats['unique_values'],
                    'Min': f"{stats['min']:.4f}",
                    'Max': f"{stats['max']:.4f}",
                    'Mean': f"{stats['mean']:.4f}",
                    'Median': f"{stats['median']:.4f}",
                    'Std_Dev': f"{stats['std_dev']:.4f}",
                    'Skewness': f"{stats['skewness']:.4f}"
                })
            else:  # Categorical column
                summary_data.append({
                    'Column': column,
                    'Data_Type': stats['data_type'],
                    'Missing_Values': stats['missing_values'],
                    'Missing_%': f"{stats['missing_percentage']:.4f}%",
                    'Unique_Values': stats['unique_values'],
                    'Min': 'N/A',
                    'Max': 'N/A',
                    'Mean': 'N/A',
                    'Median': 'N/A',
                    'Std_Dev': 'N/A',
                    'Skewness': 'N/A'
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display the table
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('../analysis/statistical/statistical_summary.csv', index=False)
        print(f"\nSummary table saved as 'statistical_summary.csv'")
        
        return summary_df
    
    def create_visualizations(self):
        """Create statistical visualizations."""
        print("\n" + "=" * 80)
        print("CREATING STATISTICAL VISUALIZATIONS")
        print("=" * 80)
        
        # Get numerical columns only
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # 1. Missing values heatmap
        plt.figure(figsize=(15, 8))
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', 
                    fontsize=20, fontweight='bold', transform=plt.gca().transAxes)
            plt.title('Missing Values Analysis - No Missing Values', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../results/data_quality/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Descriptive statistics comparison
        if len(numerical_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            
            # Mean vs Median comparison
            means = [self.stats_summary[col]['mean'] for col in numerical_cols]
            medians = [self.stats_summary[col]['median'] for col in numerical_cols]
            
            axes[0, 0].scatter(means, medians, alpha=0.7, s=100)
            axes[0, 0].plot([min(means), max(means)], [min(means), max(means)], 'r--', alpha=0.8)
            axes[0, 0].set_xlabel('Mean')
            axes[0, 0].set_ylabel('Median')
            axes[0, 0].set_title('Mean vs Median Comparison')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add labels for each point
            for i, col in enumerate(numerical_cols):
                axes[0, 0].annotate(col, (means[i], medians[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Standard deviation distribution
            std_devs = [self.stats_summary[col]['std_dev'] for col in numerical_cols]
            axes[0, 1].bar(range(len(numerical_cols)), std_devs, alpha=0.7)
            axes[0, 1].set_xlabel('Features')
            axes[0, 1].set_ylabel('Standard Deviation')
            axes[0, 1].set_title('Standard Deviation by Feature')
            axes[0, 1].set_xticks(range(len(numerical_cols)))
            axes[0, 1].set_xticklabels(numerical_cols, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Skewness distribution
            skewness_vals = [self.stats_summary[col]['skewness'] for col in numerical_cols]
            axes[1, 0].bar(range(len(numerical_cols)), skewness_vals, alpha=0.7, color='orange')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 0].set_xlabel('Features')
            axes[1, 0].set_ylabel('Skewness')
            axes[1, 0].set_title('Skewness by Feature (0 = Normal)')
            axes[1, 0].set_xticks(range(len(numerical_cols)))
            axes[1, 0].set_xticklabels(numerical_cols, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Range (Max - Min) distribution
            ranges = [self.stats_summary[col]['max'] - self.stats_summary[col]['min'] for col in numerical_cols]
            axes[1, 1].bar(range(len(numerical_cols)), ranges, alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Range (Max - Min)')
            axes[1, 1].set_title('Value Range by Feature')
            axes[1, 1].set_xticks(range(len(numerical_cols)))
            axes[1, 1].set_xticklabels(numerical_cols, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('../results/statistical/statistical_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Data type distribution
        plt.figure(figsize=(10, 6))
        data_types = self.df.dtypes.value_counts()
        plt.pie(data_types.values, labels=data_types.index, autopct='%1.1f%%', startangle=90)
        plt.title('Data Type Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../results/statistical/data_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive statistical report."""
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS REPORT")
        print("=" * 80)
        
        # Dataset overview
        print(f"\nDATASET OVERVIEW:")
        print(f"Total samples: {len(self.df):,}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"Total cells: {len(self.df) * len(self.df.columns):,}")
        
        # Missing values summary
        total_missing = self.df.isnull().sum().sum()
        print(f"\nMISSING VALUES SUMMARY:")
        print(f"Total missing values: {total_missing:,}")
        print(f"Missing percentage: {(total_missing / (len(self.df) * len(self.df.columns))) * 100:.4f}%")
        
        if total_missing == 0:
            print("Excellent data quality - No missing values!")
        
        # Data types summary
        print(f"\nDATA TYPES SUMMARY:")
        data_type_counts = self.df.dtypes.value_counts()
        for dtype, count in data_type_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Numerical columns analysis
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\nNUMERICAL COLUMNS ANALYSIS:")
        print(f"Number of numerical columns: {len(numerical_cols)}")
        
        if len(numerical_cols) > 0:
            # Overall statistics for numerical columns
            all_means = [self.stats_summary[col]['mean'] for col in numerical_cols]
            all_stds = [self.stats_summary[col]['std_dev'] for col in numerical_cols]
            
            print(f"Average mean across numerical columns: {np.mean(all_means):.4f}")
            print(f"Average standard deviation: {np.mean(all_stds):.4f}")
            
            # Identify columns with high variability
            high_var_cols = [col for col in numerical_cols if self.stats_summary[col]['std_dev'] > np.mean(all_stds) * 2]
            if high_var_cols:
                print(f"Columns with high variability: {high_var_cols}")
        
        # Categorical columns analysis
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        print(f"\nCATEGORICAL COLUMNS ANALYSIS:")
        print(f"Number of categorical columns: {len(categorical_cols)}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if total_missing == 0:
            print("Data quality is excellent - no preprocessing needed for missing values")
        
        if len(numerical_cols) > 0:
            skewed_cols = [col for col in numerical_cols if abs(self.stats_summary[col]['skewness']) > 1]
            if skewed_cols:
                print(f"Consider log transformation for skewed columns: {skewed_cols}")
            else:
                print("No highly skewed numerical columns detected")
        
        print("Dataset is ready for machine learning analysis")
        
        return {
            'total_samples': len(self.df),
            'total_features': len(self.df.columns),
            'missing_values': total_missing,
            'numerical_columns': len(numerical_cols),
            'categorical_columns': len(categorical_cols)
        }

def main():
    """Main function to run the complete statistical analysis."""
    # Initialize analyzer
    analyzer = StatisticalAnalyzer('../data/heart_disease_risk_dataset_earlymed.csv')
    
    # Run complete statistical analysis
    analyzer.load_data()
    analyzer.comprehensive_statistical_summary()
    summary_table = analyzer.create_summary_table()
    analyzer.create_visualizations()
    report = analyzer.generate_report()
    
    print(f"\nStatistical analysis completed successfully!")
    print(f"Generated files:")
    print(f"  - statistical_summary.csv")
    print(f"  - missing_values_heatmap.png")
    print(f"  - statistical_comparison.png")
    print(f"  - data_type_distribution.png")

if __name__ == "__main__":
    main()
