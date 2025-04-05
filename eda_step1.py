"""
EDA Step 1: Summary Statistics Analysis
This script performs initial exploratory data analysis by:
1. Loading and examining the dataset
2. Identifying variable types automatically
3. Computing summary statistics for numerical variables
4. Computing mode for categorical variables
5. Identifying potential unusual variables
6. Investigating zero values
7. Creating binary flags for zero values
8. Saving results to output directory
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path('eda_step1_output')
output_dir.mkdir(exist_ok=True)

def load_data():
    """Load the cleaned customer dataset"""
    return pd.read_csv('Customer_cleaned.csv')

def create_binary_flags(df):
    """
    Create binary flags for all numeric variables with zero values
    Returns the dataframe with new binary flag columns
    """
    df_with_flags = df.copy()
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Create binary flag column
            flag_name = f'has_{column.lower()}'
            df_with_flags[flag_name] = (df[column] > 0).astype(int)
    
    return df_with_flags

def identify_variable_types(df):
    """
    Automatically identify variable types based on patterns
    Returns a dictionary with variable types
    """
    var_types = {}
    
    # List of percentage variables that should be treated as numerical
    percentage_vars = ['PctSessionPurchase', 'PctSessionClickDiscount', 'PctSessionPurchaseDiscount']
    
    for column in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Always treat percentage variables as numerical
            if column in percentage_vars:
                var_types[column] = 'numerical'
            else:
                # Check if it might be categorical despite being numeric
                unique_ratio = df[column].nunique() / len(df)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    var_types[column] = 'categorical_numeric'
                else:
                    var_types[column] = 'numerical'
        else:
            var_types[column] = 'categorical'
    
    return var_types

def compute_summary_stats(df, var_types):
    """
    Compute summary statistics based on variable types
    """
    summary_stats = {}
    
    for column, var_type in var_types.items():
        if var_type == 'numerical':
            stats = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'percentiles': df[column].quantile([0.25, 0.5, 0.75]).to_dict()
            }
            summary_stats[column] = stats
        else:  # categorical or categorical_numeric
            mode_value = df[column].mode().iloc[0]
            summary_stats[column] = {'mode': mode_value}
    
    return summary_stats

def identify_unusual_variables(df, var_types):
    """
    Identify potentially unusual variables that might need attention
    """
    unusual_vars = []
    
    for column, var_type in var_types.items():
        if var_type == 'categorical_numeric':
            unusual_vars.append({
                'column': column,
                'type': 'categorical_numeric',
                'unique_values': df[column].nunique(),
                'value_counts': df[column].value_counts().head().to_dict()
            })
    
    return unusual_vars

def investigate_zero_values(df):
    """
    Investigate zero values in the dataset
    Returns a dictionary with zero value analysis for each numeric column
    """
    zero_analysis = {}
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            zero_count = (df[column] == 0).sum()
            zero_percentage = (zero_count / len(df)) * 100
            
            # For columns with significant zero values, analyze the distribution
            if zero_percentage > 0:
                non_zero_stats = df[df[column] != 0][column].describe()
                
                zero_analysis[column] = {
                    'zero_count': zero_count,
                    'zero_percentage': zero_percentage,
                    'non_zero_stats': non_zero_stats.to_dict(),
                    'potential_issues': []
                }
                
                # Identify potential issues based on patterns
                if zero_percentage > 50:
                    zero_analysis[column]['potential_issues'].append(
                        "High proportion of zeros (>50%) - might indicate inactive users or data collection issues"
                    )
                if zero_percentage > 0 and zero_percentage < 5:
                    zero_analysis[column]['potential_issues'].append(
                        "Low proportion of zeros (<5%) - might be legitimate zero values"
                    )
                if df[column].nunique() < 10:
                    zero_analysis[column]['potential_issues'].append(
                        "Limited unique values - might be categorical in nature"
                    )
    
    return zero_analysis

def suggest_zero_value_treatments(zero_analysis):
    """
    Generate treatment suggestions for zero values
    """
    treatments = {}
    
    for column, analysis in zero_analysis.items():
        treatments[column] = {
            'current_state': f"Zeros: {analysis['zero_count']} ({analysis['zero_percentage']:.2f}%)",
            'suggested_treatments': []
        }
        
        # Suggest treatments based on patterns
        if analysis['zero_percentage'] > 50:
            treatments[column]['suggested_treatments'].extend([
                "1. Consider removing these users from analysis if they represent inactive accounts",
                "2. Create a separate category for inactive users",
                "3. Investigate if these are missing values that should be imputed"
            ])
        elif analysis['zero_percentage'] > 0 and analysis['zero_percentage'] < 5:
            treatments[column]['suggested_treatments'].extend([
                "1. Keep as legitimate zero values",
                "2. Consider log transformation (adding small constant) for modeling"
            ])
        else:
            treatments[column]['suggested_treatments'].extend([
                "1. Investigate business context of zero values",
                "2. Consider median imputation if zeros represent missing values",
                "3. Create binary flag for zero/non-zero values"
            ])
    
    return treatments

def main():
    # Load data
    df = load_data()
    
    # Identify variable types
    var_types = identify_variable_types(df)
    
    # Compute summary statistics
    summary_stats = compute_summary_stats(df, var_types)
    
    # Identify unusual variables
    unusual_vars = identify_unusual_variables(df, var_types)
    
    # Investigate zero values
    zero_analysis = investigate_zero_values(df)
    zero_treatments = suggest_zero_value_treatments(zero_analysis)
    
    # Create binary flags
    df_with_flags = create_binary_flags(df)
    
    # Save results
    with open(output_dir / 'variable_types.txt', 'w') as f:
        f.write("Variable Types:\n")
        for col, type_ in var_types.items():
            f.write(f"{col}: {type_}\n")
    
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write("Summary Statistics:\n")
        for col, stats in summary_stats.items():
            f.write(f"\n{col}:\n")
            for stat, value in stats.items():
                f.write(f"{stat}: {value}\n")
    
    with open(output_dir / 'unusual_variables.txt', 'w') as f:
        f.write("Potentially Unusual Variables:\n")
        for var in unusual_vars:
            f.write(f"\n{var['column']}:\n")
            f.write(f"Type: {var['type']}\n")
            f.write(f"Number of unique values: {var['unique_values']}\n")
            f.write("Top 5 value counts:\n")
            for val, count in var['value_counts'].items():
                f.write(f"{val}: {count}\n")
    
    with open(output_dir / 'zero_value_analysis.txt', 'w') as f:
        f.write("Zero Value Analysis:\n")
        for col, analysis in zero_analysis.items():
            f.write(f"\n{col}:\n")
            f.write(f"Zero Count: {analysis['zero_count']}\n")
            f.write(f"Zero Percentage: {analysis['zero_percentage']:.2f}%\n")
            f.write("Non-zero Statistics:\n")
            for stat, value in analysis['non_zero_stats'].items():
                f.write(f"{stat}: {value}\n")
            f.write("Potential Issues:\n")
            for issue in analysis['potential_issues']:
                f.write(f"- {issue}\n")
    
    with open(output_dir / 'zero_value_treatments.txt', 'w') as f:
        f.write("Suggested Treatments for Zero Values:\n")
        for col, treatment in zero_treatments.items():
            f.write(f"\n{col}:\n")
            f.write(f"Current State: {treatment['current_state']}\n")
            f.write("Suggested Treatments:\n")
            for suggestion in treatment['suggested_treatments']:
                f.write(f"- {suggestion}\n")
    
    # Save the processed dataset with binary flags
    df_with_flags.to_csv(output_dir / 'customer_data_with_flags.csv', index=False)
    
    # Save binary flag summary
    with open(output_dir / 'binary_flags_summary.txt', 'w') as f:
        f.write("Binary Flags Summary:\n")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                flag_name = f'has_{col.lower()}'
                flag_counts = df_with_flags[flag_name].value_counts()
                f.write(f"\n{flag_name}:\n")
                f.write(f"1 (Non-zero): {flag_counts[1]} ({flag_counts[1]/len(df)*100:.2f}%)\n")
                f.write(f"0 (Zero): {flag_counts[0]} ({flag_counts[0]/len(df)*100:.2f}%)\n")

if __name__ == "__main__":
    main() 