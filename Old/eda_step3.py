"""
EDA Step 3: Correlation and Relationships Analysis
This script performs correlation analysis by:
1. Loading the transformed dataset from step 2
2. Computing pairwise correlations
3. Generating correlation heatmaps
4. Calculating VIF scores
5. Identifying significant relationships
6. Suggesting feature selection and engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations

# Create output directory if it doesn't exist
output_dir = Path('eda_step3_output')
output_dir.mkdir(exist_ok=True)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the transformed dataset from step 2"""
    return pd.read_csv('eda_step2_output/customer_data_transformed.csv')

def compute_correlations(df):
    """
    Compute different types of correlations:
    - Pearson (linear)
    - Spearman (monotonic)
    - Kendall (ordinal)
    """
    # Select numeric columns (excluding binary flags)
    numeric_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) 
                   and not col.startswith('has_')]
    
    correlations = {
        'pearson': df[numeric_cols].corr(method='pearson'),
        'spearman': df[numeric_cols].corr(method='spearman'),
        'kendall': df[numeric_cols].corr(method='kendall')
    }
    
    return correlations, numeric_cols

def plot_correlation_heatmaps(correlations):
    """Generate heatmaps for each correlation type"""
    for corr_type, corr_matrix in correlations.items():
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title(f'{corr_type.capitalize()} Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / f'correlation_heatmap_{corr_type}.png')
        plt.close()

def calculate_vif(df, numeric_cols):
    """Calculate Variance Inflation Factor for each feature"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i)
                       for i in range(len(numeric_cols))]
    return vif_data

def identify_significant_correlations(correlations, p_value_threshold=0.05):
    """
    Identify statistically significant correlations
    Returns dictionary of significant correlations for each correlation type
    """
    significant_correlations = {}
    
    for corr_type, corr_matrix in correlations.items():
        significant_pairs = []
        
        for i, j in combinations(corr_matrix.index, 2):
            if corr_type == 'pearson':
                # Calculate p-value for Pearson correlation
                r = corr_matrix.loc[i, j]
                n = len(correlations[corr_type])
                t = r * np.sqrt((n-2)/(1-r**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t), n-2))
            else:
                # For Spearman and Kendall, use their respective correlation tests
                if corr_type == 'spearman':
                    r, p_value = stats.spearmanr(correlations[corr_type][i], correlations[corr_type][j])
                else:  # kendall
                    r, p_value = stats.kendalltau(correlations[corr_type][i], correlations[corr_type][j])
            
            if p_value < p_value_threshold:
                significant_pairs.append({
                    'var1': i,
                    'var2': j,
                    'correlation': corr_matrix.loc[i, j],
                    'p_value': p_value
                })
        
        significant_correlations[corr_type] = significant_pairs
    
    return significant_correlations

def suggest_feature_engineering(correlations, significant_correlations, vif_data):
    """
    Suggest feature selection and engineering based on:
    1. High correlations (potential redundancy)
    2. High VIF scores (multicollinearity)
    3. Strong relationships (potential interaction terms)
    """
    suggestions = {
        'drop_features': [],
        'interaction_terms': [],
        'high_vif_features': []
    }
    
    # Identify features with high VIF (> 5)
    high_vif = vif_data[vif_data['VIF'] > 5]
    suggestions['high_vif_features'] = high_vif['Feature'].tolist()
    
    # Identify highly correlated pairs (|r| > 0.7)
    for corr_type, corr_matrix in correlations.items():
        if corr_type == 'pearson':  # Use Pearson for feature selection
            for i, j in combinations(corr_matrix.index, 2):
                if abs(corr_matrix.loc[i, j]) > 0.7:
                    # Suggest dropping the feature with higher VIF
                    vif_i = vif_data[vif_data['Feature'] == i]['VIF'].values[0]
                    vif_j = vif_data[vif_data['Feature'] == j]['VIF'].values[0]
                    if vif_i > vif_j:
                        if i not in suggestions['drop_features']:
                            suggestions['drop_features'].append(i)
                    else:
                        if j not in suggestions['drop_features']:
                            suggestions['drop_features'].append(j)
    
    # Define specific interaction terms to create
    interaction_pairs = [
        ('AvgTimePerClick', 'AvgPriceClicked'),
        ('ItemsClickedPerSession', 'AvgPriceClicked'),
        ('SessionCount', 'PctSessionPurchase'),
        ('AvgTimePerSession', 'PctSessionPurchase')
    ]
    
    # Add interaction terms to suggestions
    for var1, var2 in interaction_pairs:
        suggestions['interaction_terms'].append({
            'variables': (var1, var2),
            'correlation': correlations['pearson'].loc[var1, var2]
        })
    
    return suggestions

def plot_scatter_matrix(df, numeric_cols):
    """Generate scatter matrix for numeric variables"""
    # Limit to reasonable number of plots
    if len(numeric_cols) > 6:
        # Select variables with highest correlations or most significant relationships
        pearson_corr = df[numeric_cols].corr()
        mean_corr = abs(pearson_corr).mean()
        selected_cols = mean_corr.nlargest(6).index.tolist()
    else:
        selected_cols = numeric_cols
    
    # Create scatter matrix
    fig = sns.pairplot(df[selected_cols], diag_kind='kde')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_matrix.png')
    plt.close()

def apply_feature_selection_and_engineering(df, suggestions):
    """
    Apply feature selection and engineering based on suggestions
    """
    # Create a copy of the dataframe
    processed_df = df.copy()
    
    # Drop selected features
    features_to_drop = suggestions['drop_features']
    processed_df = processed_df.drop(columns=features_to_drop)
    
    # Create interaction terms
    for term in suggestions['interaction_terms']:
        var1, var2 = term['variables']
        interaction_name = f"{var1}_{var2}_interaction"
        processed_df[interaction_name] = processed_df[var1] * processed_df[var2]
    
    return processed_df

def analyze_new_correlations(processed_df):
    """
    Analyze correlations focusing on the interaction terms
    """
    # Get correlations for the processed dataset
    correlations, numeric_cols = compute_correlations(processed_df)
    
    # Plot new correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlations['pearson'], 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Correlation Heatmap (Including Interaction Terms)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap_with_interactions.png')
    plt.close()
    
    # Analyze correlations of interaction terms
    interaction_terms = [col for col in numeric_cols if col.endswith('_interaction')]
    
    with open(output_dir / 'interaction_terms_analysis.txt', 'w') as f:
        f.write("Analysis of Interaction Terms:\n\n")
        
        for term in interaction_terms:
            f.write(f"\nCorrelations for {term}:\n")
            # Get correlations with other variables
            correlations_with_term = correlations['pearson'][term].sort_values(ascending=False)
            
            # Write top 5 strongest correlations
            f.write("Top 5 strongest correlations:\n")
            for var, corr in correlations_with_term.head().items():
                if var != term:  # Skip self-correlation
                    f.write(f"- {var}: {corr:.3f}\n")
            
            # Check for potential multicollinearity
            high_corrs = correlations_with_term[abs(correlations_with_term) > 0.7]
            if len(high_corrs) > 1:  # More than just self-correlation
                f.write("\nPotential multicollinearity (|r| > 0.7) with:\n")
                for var, corr in high_corrs.items():
                    if var != term:
                        f.write(f"- {var}: {corr:.3f}\n")
            
            f.write("\n" + "-"*50 + "\n")

def main():
    # Load data
    df = load_data()
    
    # Compute correlations
    correlations, numeric_cols = compute_correlations(df)
    
    # Plot correlation heatmaps
    plot_correlation_heatmaps(correlations)
    
    # Calculate VIF
    vif_data = calculate_vif(df, numeric_cols)
    
    # Identify significant correlations
    significant_correlations = identify_significant_correlations(correlations)
    
    # Generate suggestions for feature engineering
    suggestions = suggest_feature_engineering(correlations, significant_correlations, vif_data)
    
    # Apply feature selection and engineering
    processed_df = apply_feature_selection_and_engineering(df, suggestions)
    
    # Analyze new correlations including interaction terms
    analyze_new_correlations(processed_df)
    
    # Plot scatter matrix
    plot_scatter_matrix(df, numeric_cols)
    
    # Save results
    with open(output_dir / 'correlation_analysis.txt', 'w') as f:
        f.write("Correlation Analysis Results:\n\n")
        
        f.write("Significant Correlations:\n")
        for corr_type, sig_pairs in significant_correlations.items():
            f.write(f"\n{corr_type.capitalize()} Correlations:\n")
            for pair in sig_pairs:
                f.write(f"- {pair['var1']} vs {pair['var2']}: "
                       f"r = {pair['correlation']:.3f} (p = {pair['p_value']:.4f})\n")
        
        f.write("\nVIF Analysis:\n")
        f.write(vif_data.to_string())
        
        f.write("\n\nFeature Engineering Decisions:\n")
        f.write("\nFeatures Dropped:\n")
        for feature in suggestions['drop_features']:
            f.write(f"- {feature}\n")
        
        f.write("\nInteraction Terms Created:\n")
        for term in suggestions['interaction_terms']:
            f.write(f"- {term['variables'][0]} × {term['variables'][1]} "
                   f"(correlation: {term['correlation']:.3f})\n")
        
        f.write("\nFeatures with High VIF (> 5):\n")
        for feature in suggestions['high_vif_features']:
            vif_value = vif_data[vif_data['Feature'] == feature]['VIF'].values[0]
            f.write(f"- {feature}: VIF = {vif_value:.2f}\n")
    
    # Save the processed dataset
    processed_df.to_csv(output_dir / 'customer_data_processed.csv', index=False)
    
    # Save summary of changes
    with open(output_dir / 'feature_engineering_summary.txt', 'w') as f:
        f.write("Feature Engineering Summary:\n\n")
        f.write("Original Features:\n")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and not col.startswith('has_'):
                f.write(f"- {col}\n")
        
        f.write("\nDropped Features:\n")
        for feature in suggestions['drop_features']:
            f.write(f"- {feature}\n")
        
        f.write("\nNew Interaction Terms:\n")
        for term in suggestions['interaction_terms']:
            var1, var2 = term['variables']
            f.write(f"- {var1}_{var2}_interaction\n")
        
        f.write("\nFinal Feature Set:\n")
        for col in processed_df.columns:
            if pd.api.types.is_numeric_dtype(processed_df[col]) and not col.startswith('has_'):
                f.write(f"- {col}\n")

if __name__ == "__main__":
    main() 
