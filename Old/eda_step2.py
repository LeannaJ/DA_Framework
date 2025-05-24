"""
EDA Step 2: Distribution Analysis
This script performs distribution analysis by:
1. Loading the dataset with binary flags from step 1
2. Generating distribution plots (histograms, density plots)
3. Calculating skewness and kurtosis
4. Suggesting transformations for skewed data
5. Saving all plots and statistics to output directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import os

# Create output directory if it doesn't exist
output_dir = Path('eda_step2_output')
output_dir.mkdir(exist_ok=True)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the dataset with binary flags from step 1"""
    return pd.read_csv('eda_step1_output/customer_data_with_flags.csv')

def calculate_distribution_stats(df):
    """
    Calculate distribution statistics for numerical variables
    Returns a dictionary with statistics for each variable
    """
    stats_dict = {}
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and not column.startswith('has_'):
            stats_dict[column] = {
                'skewness': stats.skew(df[column]),
                'kurtosis': stats.kurtosis(df[column]),
                'shapiro_test': stats.shapiro(df[column]),
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std()
            }
    
    return stats_dict

def plot_distributions(df, stats_dict):
    """
    Generate distribution plots for numerical variables
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and not column.startswith('has_'):
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram with KDE
            sns.histplot(data=df, x=column, kde=True, ax=ax1)
            ax1.set_title(f'Histogram of {column}')
            ax1.set_xlabel(column)
            ax1.set_ylabel('Count')
            
            # Q-Q plot
            stats.probplot(df[column], dist="norm", plot=ax2)
            ax2.set_title(f'Q-Q Plot of {column}')
            
            # Add distribution statistics as text
            stats_text = f"Skewness: {stats_dict[column]['skewness']:.2f}\n"
            stats_text += f"Kurtosis: {stats_dict[column]['kurtosis']:.2f}\n"
            stats_text += f"Shapiro p-value: {stats_dict[column]['shapiro_test'][1]:.4f}"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=10)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_dir / f'distribution_{column}.png')
            plt.close()

def suggest_transformations(stats_dict):
    """
    Suggest transformations based on distribution statistics
    Only suggest log transformations for highly skewed variables
    """
    suggestions = {}
    
    for column, stats in stats_dict.items():
        suggestions[column] = {
            'current_distribution': 'normal' if abs(stats['skewness']) < 1.5 else 'highly skewed',
            'skewness': stats['skewness'],
            'kurtosis': stats['kurtosis'],
            'suggested_transformations': []
        }
        
        # Only suggest transformations for highly skewed variables
        if abs(stats['skewness']) > 1.5:
            suggestions[column]['suggested_transformations'].append(
                "Log transformation (log(x + 1))"
            )
    
    return suggestions

def apply_transformations(df, stats_dict):
    """
    Apply log transformations only to highly skewed variables (skewness > 1.5)
    """
    transformed_df = df.copy()
    transformed_vars = []
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and not column.startswith('has_'):
            stats = stats_dict[column]
            
            # Only transform highly skewed variables
            if abs(stats['skewness']) > 1.5:
                # Log transformation
                transformed_df[f'{column}_log'] = np.log1p(df[column])
                transformed_vars.append(column)
    
    # Save the list of transformed variables
    with open(output_dir / 'transformed_variables.txt', 'w') as f:
        f.write("Variables that were log-transformed (skewness > 1.5):\n\n")
        for var in transformed_vars:
            f.write(f"{var}: skewness = {stats_dict[var]['skewness']:.2f}\n")
    
    return transformed_df

def plot_transformed_distributions(df, transformed_df, stats_dict):
    """
    Plot distributions of transformed variables
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and not column.startswith('has_'):
            column_stats = stats_dict[column]
            
            # Only plot highly skewed variables
            if abs(column_stats['skewness']) > 1.5:
                # Create figure with subplots
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                fig.suptitle(f'Distribution Analysis for {column}')
                
                # Original distribution
                sns.histplot(data=df, x=column, kde=True, ax=axes[0])
                axes[0].set_title('Original Distribution')
                
                # Log transformation
                if f'{column}_log' in transformed_df.columns:
                    sns.histplot(data=transformed_df, x=f'{column}_log', kde=True, ax=axes[1])
                    axes[1].set_title('Log Transformation')
                
                # Q-Q plot of original data
                stats.probplot(df[column].values, dist="norm", plot=axes[2])
                axes[2].set_title('Q-Q Plot (Original)')
                
                # Add skewness information
                plt.figtext(0.02, 0.02, f"Original Skewness: {column_stats['skewness']:.2f}", fontsize=10)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'transformed_distribution_{column}.png')
                plt.close()

def main():
    # Load data
    df = load_data()
    
    # Calculate distribution statistics
    stats_dict = calculate_distribution_stats(df)
    
    # Plot original distributions
    plot_distributions(df, stats_dict)
    
    # Generate transformation suggestions
    suggestions = suggest_transformations(stats_dict)
    
    # Apply transformations
    transformed_df = apply_transformations(df, stats_dict)
    
    # Plot transformed distributions
    plot_transformed_distributions(df, transformed_df, stats_dict)
    
    # Save statistics and suggestions
    with open(output_dir / 'distribution_statistics.txt', 'w') as f:
        f.write("Distribution Statistics:\n")
        for column, stats in stats_dict.items():
            f.write(f"\n{column}:\n")
            f.write(f"Skewness: {stats['skewness']:.2f}\n")
            f.write(f"Kurtosis: {stats['kurtosis']:.2f}\n")
            f.write(f"Shapiro-Wilk p-value: {stats['shapiro_test'][1]:.4f}\n")
            f.write(f"Mean: {stats['mean']:.2f}\n")
            f.write(f"Median: {stats['median']:.2f}\n")
            f.write(f"Standard Deviation: {stats['std']:.2f}\n")
    
    with open(output_dir / 'transformation_suggestions.txt', 'w') as f:
        f.write("Transformation Suggestions:\n")
        for column, suggestion in suggestions.items():
            f.write(f"\n{column}:\n")
            f.write(f"Current Distribution: {suggestion['current_distribution']}\n")
            f.write(f"Skewness: {suggestion['skewness']:.2f}\n")
            f.write(f"Kurtosis: {suggestion['kurtosis']:.2f}\n")
            f.write("Suggested Transformations:\n")
            for transform in suggestion['suggested_transformations']:
                f.write(f"- {transform}\n")
    
    # Save transformed dataset
    transformed_df.to_csv(output_dir / 'customer_data_transformed.csv', index=False)

if __name__ == "__main__":
    main() 
