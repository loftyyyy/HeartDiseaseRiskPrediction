"""
IQR Outlier Analysis for Heart Disease Risk Dataset
Comprehensive outlier detection using Interquartile Range method for all columns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class IQROutlierAnalyzer:
    def __init__(self, data_path):
        """Initialize the IQR analyzer with the dataset path."""
        self.data_path = data_path
        self.df = None
        self.outlier_results = {}
        self.outlier_summary = {}
        
    def load_data(self):
        """Load the dataset."""
        print("=" * 80)
        print("IQR OUTLIER ANALYSIS FOR HEART DISEASE RISK DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def calculate_iqr_outliers(self, column):
        """Calculate IQR outliers for a specific column."""
        data = self.df[column].dropna()  # Remove any NaN values
        
        # Calculate quartiles
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Calculate outlier statistics
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        return {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'outliers': outliers,
            'min_value': data.min(),
            'max_value': data.max(),
            'mean_value': data.mean(),
            'std_value': data.std()
        }
    
    def analyze_all_columns(self):
        """Perform IQR analysis on all columns."""
        print("\n" + "=" * 80)
        print("IQR OUTLIER ANALYSIS FOR ALL COLUMNS")
        print("=" * 80)
        
        for column in self.df.columns:
            print(f"\n{'='*60}")
            print(f"IQR ANALYSIS FOR COLUMN: {column}")
            print(f"{'='*60}")
            
            # Calculate IQR outliers
            iqr_stats = self.calculate_iqr_outliers(column)
            
            # Display results
            print(f"Data Type: {self.df[column].dtype}")
            print(f"Total Values: {len(self.df[column]):,}")
            print(f"Missing Values: {self.df[column].isnull().sum():,}")
            
            print(f"\nQUARTILE ANALYSIS:")
            print(f"Q1 (25th percentile): {iqr_stats['Q1']:.4f}")
            print(f"Q3 (75th percentile): {iqr_stats['Q3']:.4f}")
            print(f"IQR (Q3 - Q1): {iqr_stats['IQR']:.4f}")
            
            print(f"\nOUTLIER BOUNDS:")
            print(f"Lower Bound (Q1 - 1.5*IQR): {iqr_stats['lower_bound']:.4f}")
            print(f"Upper Bound (Q3 + 1.5*IQR): {iqr_stats['upper_bound']:.4f}")
            
            print(f"\nDATA RANGE:")
            print(f"Minimum Value: {iqr_stats['min_value']:.4f}")
            print(f"Maximum Value: {iqr_stats['max_value']:.4f}")
            print(f"Mean Value: {iqr_stats['mean_value']:.4f}")
            print(f"Standard Deviation: {iqr_stats['std_value']:.4f}")
            
            print(f"\nOUTLIER ANALYSIS:")
            print(f"Number of Outliers: {iqr_stats['outlier_count']:,}")
            print(f"Percentage of Outliers: {iqr_stats['outlier_percentage']:.4f}%")
            
            # Show outlier values if any
            if iqr_stats['outlier_count'] > 0:
                print(f"\nOutlier Values:")
                outlier_values = sorted(iqr_stats['outliers'].unique())
                if len(outlier_values) <= 20:  # Show all if 20 or fewer
                    for val in outlier_values:
                        count = (iqr_stats['outliers'] == val).sum()
                        print(f"  {val}: {count} occurrences")
                else:  # Show summary if too many
                    print(f"  {len(outlier_values)} unique outlier values")
                    print(f"  Range: {min(outlier_values):.4f} to {max(outlier_values):.4f}")
            else:
                print("No outliers detected!")
            
            # Interpretation
            print(f"\nINTERPRETATION:")
            if iqr_stats['outlier_count'] == 0:
                print("No outliers detected - data quality is excellent")
                print("All values are within expected range")
                print("No outlier treatment required")
            else:
                print(f"{iqr_stats['outlier_count']} outliers detected")
                if iqr_stats['outlier_percentage'] < 5:
                    print("Low outlier percentage - acceptable")
                elif iqr_stats['outlier_percentage'] < 10:
                    print("Moderate outlier percentage - review recommended")
                else:
                    print("High outlier percentage - investigation required")
            
            # Store results
            self.outlier_results[column] = iqr_stats
            
            # Create summary entry
            self.outlier_summary[column] = {
                'data_type': str(self.df[column].dtype),
                'total_values': len(self.df[column]),
                'missing_values': self.df[column].isnull().sum(),
                'Q1': iqr_stats['Q1'],
                'Q3': iqr_stats['Q3'],
                'IQR': iqr_stats['IQR'],
                'lower_bound': iqr_stats['lower_bound'],
                'upper_bound': iqr_stats['upper_bound'],
                'min_value': iqr_stats['min_value'],
                'max_value': iqr_stats['max_value'],
                'mean_value': iqr_stats['mean_value'],
                'std_value': iqr_stats['std_value'],
                'outlier_count': iqr_stats['outlier_count'],
                'outlier_percentage': iqr_stats['outlier_percentage']
            }
    
    def create_summary_table(self):
        """Create a comprehensive summary table."""
        print("\n" + "=" * 80)
        print("IQR OUTLIER ANALYSIS SUMMARY TABLE")
        print("=" * 80)
        
        # Create summary DataFrame
        summary_data = []
        
        for column, stats in self.outlier_summary.items():
            summary_data.append({
                'Column': column,
                'Data_Type': stats['data_type'],
                'Total_Values': stats['total_values'],
                'Missing_Values': stats['missing_values'],
                'Q1': f"{stats['Q1']:.4f}",
                'Q3': f"{stats['Q3']:.4f}",
                'IQR': f"{stats['IQR']:.4f}",
                'Lower_Bound': f"{stats['lower_bound']:.4f}",
                'Upper_Bound': f"{stats['upper_bound']:.4f}",
                'Min_Value': f"{stats['min_value']:.4f}",
                'Max_Value': f"{stats['max_value']:.4f}",
                'Outlier_Count': stats['outlier_count'],
                'Outlier_%': f"{stats['outlier_percentage']:.4f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display the table
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('../analysis/data_quality/iqr_outlier_analysis_summary.csv', index=False)
        print(f"\nSummary table saved as 'iqr_outlier_analysis_summary.csv'")
        
        return summary_df
    
    def create_visualizations(self):
        """Create IQR analysis visualizations."""
        print("\n" + "=" * 80)
        print("CREATING IQR OUTLIER VISUALIZATIONS")
        print("=" * 80)
        
        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # 1. Outlier count visualization
        plt.figure(figsize=(15, 8))
        outlier_counts = [self.outlier_summary[col]['outlier_count'] for col in numerical_cols]
        outlier_percentages = [self.outlier_summary[col]['outlier_percentage'] for col in numerical_cols]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Outlier counts
        bars1 = ax1.bar(range(len(numerical_cols)), outlier_counts, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Number of Outliers')
        ax1.set_title('Number of Outliers by Feature (IQR Method)')
        ax1.set_xticks(range(len(numerical_cols)))
        ax1.set_xticklabels(numerical_cols, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Outlier percentages
        bars2 = ax2.bar(range(len(numerical_cols)), outlier_percentages, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Outlier Percentage (%)')
        ax2.set_title('Outlier Percentage by Feature (IQR Method)')
        ax2.set_xticks(range(len(numerical_cols)))
        ax2.set_xticklabels(numerical_cols, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('../results/data_quality/iqr_outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Box plots for numerical columns
        if len(numerical_cols) > 0:
            n_cols = min(4, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    # Create box plot
                    box_data = self.df[col].dropna()
                    axes[i].boxplot(box_data, patch_artist=True, 
                                   boxprops=dict(facecolor='lightblue', alpha=0.7))
                    axes[i].set_title(f'{col}\nOutliers: {self.outlier_summary[col]["outlier_count"]}')
                    axes[i].set_ylabel('Value')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('../results/data_quality/iqr_boxplots.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self):
        """Generate a comprehensive IQR outlier analysis report."""
        print("\n" + "=" * 80)
        print("IQR OUTLIER ANALYSIS REPORT")
        print("=" * 80)
        
        # Overall statistics
        total_outliers = sum([stats['outlier_count'] for stats in self.outlier_summary.values()])
        total_values = sum([stats['total_values'] for stats in self.outlier_summary.values()])
        overall_outlier_percentage = (total_outliers / total_values) * 100
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Total data points analyzed: {total_values:,}")
        print(f"Total outliers detected: {total_outliers:,}")
        print(f"Overall outlier percentage: {overall_outlier_percentage:.4f}%")
        
        # Columns with outliers
        columns_with_outliers = [col for col, stats in self.outlier_summary.items() 
                                if stats['outlier_count'] > 0]
        columns_without_outliers = [col for col, stats in self.outlier_summary.items() 
                                   if stats['outlier_count'] == 0]
        
        print(f"\nCOLUMNS WITH OUTLIERS: {len(columns_with_outliers)}")
        for col in columns_with_outliers:
            outlier_count = self.outlier_summary[col]['outlier_count']
            outlier_pct = self.outlier_summary[col]['outlier_percentage']
            print(f"  {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
        
        print(f"\nCOLUMNS WITHOUT OUTLIERS: {len(columns_without_outliers)}")
        for col in columns_without_outliers:
            print(f"  {col}: No outliers detected")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if total_outliers == 0:
            print("EXCELLENT DATA QUALITY - No outliers detected in any column")
            print("No outlier treatment required")
            print("Dataset is ready for machine learning analysis")
        else:
            print("Outliers detected - consider the following:")
            for col in columns_with_outliers:
                outlier_pct = self.outlier_summary[col]['outlier_percentage']
                if outlier_pct < 5:
                    print(f"  {col}: Low outlier percentage ({outlier_pct:.2f}%) - acceptable")
                elif outlier_pct < 10:
                    print(f"  {col}: Moderate outlier percentage ({outlier_pct:.2f}%) - review recommended")
                else:
                    print(f"  {col}: High outlier percentage ({outlier_pct:.2f}%) - investigation required")
        
        return {
            'total_outliers': total_outliers,
            'total_values': total_values,
            'overall_outlier_percentage': overall_outlier_percentage,
            'columns_with_outliers': columns_with_outliers,
            'columns_without_outliers': columns_without_outliers
        }

def main():
    """Main function to run the complete IQR outlier analysis."""
    # Initialize analyzer
    analyzer = IQROutlierAnalyzer('../data/heart_disease_risk_dataset_earlymed.csv')
    
    # Run complete IQR analysis
    analyzer.load_data()
    analyzer.analyze_all_columns()
    summary_table = analyzer.create_summary_table()
    analyzer.create_visualizations()
    report = analyzer.generate_report()
    
    print(f"\nIQR outlier analysis completed successfully!")
    print(f"Generated files:")
    print(f"  - iqr_outlier_analysis_summary.csv")
    print(f"  - iqr_outlier_analysis.png")
    print(f"  - iqr_boxplots.png")

if __name__ == "__main__":
    main()
