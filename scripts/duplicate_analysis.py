"""
Analysis of Duplicate Rows in Heart Disease Dataset
"""

import pandas as pd
import numpy as np

def analyze_duplicates():
    """Analyze duplicate rows in the dataset."""
    print("=" * 60)
    print("DUPLICATE ROWS ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('../data/heart_disease_risk_dataset_earlymed.csv')
    
    # Find duplicates
    duplicate_mask = df.duplicated(keep=False)
    duplicates = df[duplicate_mask]
    unique_rows = df[~duplicate_mask]
    
    print(f"Total rows: {len(df):,}")
    print(f"Unique rows: {len(unique_rows):,}")
    print(f"Duplicate rows: {len(duplicates):,}")
    print(f"Percentage of duplicates: {(len(duplicates)/len(df))*100:.2f}%")
    
    # Analyze duplicate patterns
    print(f"\n=== DUPLICATE PATTERN ANALYSIS ===")
    
    # Group duplicates by their values
    duplicate_groups = df.groupby(list(df.columns)).size().reset_index(name='count')
    duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1]
    
    print(f"Number of unique duplicate patterns: {len(duplicate_groups)}")
    print(f"Most frequent duplicate pattern appears: {duplicate_groups['count'].max()} times")
    print(f"Average frequency of duplicate patterns: {duplicate_groups['count'].mean():.2f}")
    
    # Show top duplicate patterns
    print(f"\n=== TOP 10 MOST FREQUENT DUPLICATE PATTERNS ===")
    top_duplicates = duplicate_groups.nlargest(10, 'count')
    for i, (_, row) in enumerate(top_duplicates.iterrows(), 1):
        print(f"{i:2d}. Pattern appears {row['count']} times:")
        print(f"    Heart_Risk: {row['Heart_Risk']}, Age: {row['Age']}, Gender: {row['Gender']}")
        print(f"    Symptoms: Chest_Pain={row['Chest_Pain']}, Shortness_of_Breath={row['Shortness_of_Breath']}")
        print()
    
    # Analyze duplicates by target variable
    print(f"=== DUPLICATES BY TARGET VARIABLE ===")
    duplicate_target_dist = duplicates['Heart_Risk'].value_counts().sort_index()
    print("Duplicate rows distribution:")
    for target_val, count in duplicate_target_dist.items():
        print(f"  Heart_Risk {target_val}: {count:,} rows ({count/len(duplicates)*100:.1f}%)")
    
    # Check if duplicates maintain class balance
    print(f"\n=== CLASS BALANCE IN DUPLICATES ===")
    original_balance = df['Heart_Risk'].value_counts()
    duplicate_balance = duplicates['Heart_Risk'].value_counts()
    
    print("Original dataset balance:")
    for target_val, count in original_balance.items():
        print(f"  Heart_Risk {target_val}: {count:,} rows ({count/len(df)*100:.1f}%)")
    
    print("Duplicate rows balance:")
    for target_val, count in duplicate_balance.items():
        print(f"  Heart_Risk {target_val}: {count:,} rows ({count/len(duplicates)*100:.1f}%)")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. KEEP DUPLICATES if:")
    print("   - They represent legitimate patient profiles")
    print("   - Common symptom combinations in real populations")
    print("   - Dataset is synthetic/generated")
    
    print("\n2. REMOVE DUPLICATES if:")
    print("   - They are data collection errors")
    print("   - Same patient recorded multiple times")
    print("   - You want to reduce dataset size")
    
    print("\n3. ANALYSIS OPTIONS:")
    print("   - Option A: Keep all duplicates (current approach)")
    print("   - Option B: Remove duplicates (63,755 unique rows)")
    print("   - Option C: Analyze with and without duplicates")
    
    return df, duplicates, duplicate_groups

def create_clean_dataset():
    """Create a version without duplicates."""
    print(f"\n=== CREATING CLEAN DATASET (NO DUPLICATES) ===")
    
    df = pd.read_csv('../data/heart_disease_risk_dataset_earlymed.csv')
    df_clean = df.drop_duplicates()
    
    print(f"Original dataset: {len(df):,} rows")
    print(f"Clean dataset: {len(df_clean):,} rows")
    print(f"Removed: {len(df) - len(df_clean):,} duplicate rows")
    
    # Check class balance in clean dataset
    print(f"\nClass balance in clean dataset:")
    clean_balance = df_clean['Heart_Risk'].value_counts().sort_index()
    for target_val, count in clean_balance.items():
        print(f"  Heart_Risk {target_val}: {count:,} rows ({count/len(df_clean)*100:.1f}%)")
    
    # Save clean dataset
    df_clean.to_csv('../data/heart_disease_risk_dataset_clean.csv', index=False)
    print(f"\nClean dataset saved as 'heart_disease_risk_dataset_clean.csv'")
    
    return df_clean

if __name__ == "__main__":
    df, duplicates, duplicate_groups = analyze_duplicates()
    
    # Ask user what they want to do
    print(f"\n" + "="*60)
    print("WHAT WOULD YOU LIKE TO DO?")
    print("="*60)
    print("1. Keep duplicates (recommended for ML)")
    print("2. Remove duplicates and create clean dataset")
    print("3. Compare both approaches")
    
    # For now, let's create the clean dataset for comparison
    df_clean = create_clean_dataset()
