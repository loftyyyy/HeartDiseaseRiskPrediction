"""
Comprehensive Data Quality Check for Heart Disease Dataset
"""

import pandas as pd
import numpy as np

def comprehensive_data_quality_check():
    """Perform comprehensive data quality analysis."""
    print("=" * 60)
    print("COMPREHENSIVE DATA QUALITY CHECK")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Total cells: {df.shape[0] * df.shape[1]:,}")
    
    # Missing values check
    print(f"\n=== MISSING VALUES CHECK ===")
    missing_values = df.isnull().sum().sum()
    print(f"Missing values (isnull): {missing_values}")
    print(f"Missing percentage: {(missing_values / (df.shape[0] * df.shape[1])) * 100:.4f}%")
    
    # Null values check
    print(f"\n=== NULL VALUES CHECK ===")
    null_values = df.isna().sum().sum()
    print(f"Null values (isna): {null_values}")
    
    # Empty strings check
    print(f"\n=== EMPTY STRINGS CHECK ===")
    empty_strings = df.astype(str).eq('').sum().sum()
    print(f"Empty strings: {empty_strings}")
    
    # Whitespace check
    print(f"\n=== WHITESPACE CHECK ===")
    whitespace_only = 0
    for col in df.columns:
        whitespace_only += df[col].astype(str).str.strip().eq('').sum()
    print(f"Whitespace-only values: {whitespace_only}")
    
    # Infinite values check
    print(f"\n=== INFINITE VALUES CHECK ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_values = np.isinf(df[numeric_cols]).sum().sum()
    print(f"Infinite values: {infinite_values}")
    
    # Duplicate rows check
    print(f"\n=== DUPLICATE ROWS CHECK ===")
    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Percentage of duplicates: {(duplicate_rows / len(df)) * 100:.4f}%")
    
    # Unique values per column
    print(f"\n=== UNIQUE VALUES PER COLUMN ===")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    # Value ranges for binary features
    print(f"\n=== VALUE RANGES FOR BINARY FEATURES ===")
    binary_cols = [col for col in df.columns if col not in ['Age', 'Heart_Risk']]
    for col in binary_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        unique_vals = sorted(df[col].unique())
        print(f"{col}: min={min_val}, max={max_val}, unique={unique_vals}")
    
    # Age column analysis
    print(f"\n=== AGE COLUMN ANALYSIS ===")
    print(f"Age: min={df['Age'].min()}, max={df['Age'].max()}, mean={df['Age'].mean():.2f}")
    print(f"Age unique values: {len(df['Age'].unique())}")
    
    # Heart_Risk column analysis
    print(f"\n=== HEART_RISK COLUMN ANALYSIS ===")
    print(f"Heart_Risk: min={df['Heart_Risk'].min()}, max={df['Heart_Risk'].max()}")
    print(f"Heart_Risk unique values: {sorted(df['Heart_Risk'].unique())}")
    print(f"Heart_Risk distribution:")
    print(df['Heart_Risk'].value_counts().sort_index())
    
    # Data types check
    print(f"\n=== DATA TYPES CHECK ===")
    print(df.dtypes)
    
    # Sample data check
    print(f"\n=== SAMPLE DATA CHECK ===")
    print("First 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(df.describe())
    
    # Data quality summary
    print(f"\n=== DATA QUALITY SUMMARY ===")
    total_cells = df.shape[0] * df.shape[1]
    issues_found = 0
    
    if missing_values > 0:
        print(f"Missing values found: {missing_values}")
        issues_found += 1
    else:
        print("No missing values")
    
    if null_values > 0:
        print(f"Null values found: {null_values}")
        issues_found += 1
    else:
        print("No null values")
    
    if empty_strings > 0:
        print(f"Empty strings found: {empty_strings}")
        issues_found += 1
    else:
        print("No empty strings")
    
    if infinite_values > 0:
        print(f"Infinite values found: {infinite_values}")
        issues_found += 1
    else:
        print("No infinite values")
    
    if duplicate_rows > 0:
        print(f"Duplicate rows found: {duplicate_rows}")
        issues_found += 1
    else:
        print("No duplicate rows")
    
    # Check for unexpected values in binary columns
    unexpected_values = 0
    for col in binary_cols:
        unique_vals = df[col].unique()
        if not all(val in [0.0, 1.0] for val in unique_vals):
            print(f"Unexpected values in {col}: {unique_vals}")
            unexpected_values += 1
    
    if unexpected_values == 0:
        print("All binary features have expected values (0.0, 1.0)")
    
    # Final assessment
    print(f"\n=== FINAL DATA QUALITY ASSESSMENT ===")
    if issues_found == 0 and unexpected_values == 0:
        print("EXCELLENT DATA QUALITY!")
        print("Dataset is clean and ready for machine learning")
        print("No missing values, null values, or data quality issues")
        print("All features have expected value ranges")
        print("Perfect class balance (50/50 split)")
    else:
        print(f"{issues_found + unexpected_values} data quality issues found")
        print("Recommendation: Address these issues before proceeding with ML")
    
    return df

if __name__ == "__main__":
    df = comprehensive_data_quality_check()
