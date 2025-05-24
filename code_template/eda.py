import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# === LLM_INITIAL_SETUP_START ===
# (LLM will generate code to set up initial parameters and configurations)
# Example:
# session_id = '{session_id}'
# input_file = '{input_file}'
# output_base_dir = 'eda_mixed_output'
# === LLM_INITIAL_SETUP_END ===

def create_output_directory(session_id):
    """Create output directory for mixed EDA"""
    base_dir = f'eda_mixed_output/{session_id}'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def log_message(message, log_file):
    """Helper to log messages"""
    print(message)
    log_file.write(message + '\n')

def analyze_dataset_structure(df, output_dir, log_file):
    """Analyze dataset structure
    LLM will generate:
    - Column type identification
    - Summary statistics
    - Save to log file and outputs
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Numerical summary
    summary_stats = df[num_cols].describe(percentiles=[.25, .5, .75]).transpose()
    summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')
    log_message("Saved summary statistics.", log_file)

    # Categorical mode
    if cat_cols:
        cat_modes = df[cat_cols].mode().iloc[0]
        cat_modes.to_csv(f'{output_dir}/categorical_modes.csv')
        log_message("Saved categorical modes.", log_file)

    # Unusual categoricals
    unusual_cats = [col for col in cat_cols if df[col].str.isnumeric().all()]
    with open(f'{output_dir}/unusual_categoricals.txt', 'w') as f:
        f.write("Unusual categorical columns (numeric-looking):\n")
        for col in unusual_cats:
            f.write(f"- {col}\n")
    log_message("Flagged unusual categorical columns.", log_file)

    return num_cols, cat_cols

def distribution_analysis(df, num_cols, output_dir, log_file):
    """Analyze distributions of numerical columns
    LLM will generate:
    - Histograms, density plots
    - Skewness/kurtosis analysis
    - Save figures to output
    """
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histogram + Density for {col}')
        plt.savefig(f'{output_dir}/hist_density_{col}.png')
        plt.close()

        skewness = df[col].skew()
        kurtosis = df[col].kurt()
        log_message(f"{col}: Skewness = {skewness:.2f}, Kurtosis = {kurtosis:.2f}", log_file)
        if abs(skewness) > 1:
            log_message(f"Suggestion: Apply transformation to normalize {col}", log_file)

def correlation_and_vif(df, num_cols, output_dir, log_file):
    """Analyze correlations and multicollinearity
    LLM will generate:
    - Correlation heatmap
    - VIF calculation
    - Save figures and CSVs
    """
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    log_message("Saved correlation heatmap.", log_file)

    # VIF
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols].dropna())
    vif_data = pd.DataFrame()
    vif_data["feature"] = num_cols
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    vif_data.to_csv(f'{output_dir}/vif.csv', index=False)
    log_message("Calculated and saved VIF.", log_file)

def custom_analysis(df, output_dir, log_file):
    """Custom user/LLM-driven analyses
    === LLM_EDA_CUSTOM_ANALYSIS_START ===
    # (LLM will generate: grouped visual summaries, scatterplots of top correlated pairs, domain-specific insights)
    # (Remove the following examples and replace with LLM-generated code as needed)
    # Example1: User request - Compare the distribution of tripduration by specific group
    if 'usertype' in df.columns and 'tripduration' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='usertype', y='tripduration', data=df)
        plt.title('Trip Duration by User Type')
        plt.savefig(f'{output_dir}/tripduration_by_usertype.png')
        plt.close()
        log_message("Generated boxplot for tripduration by usertype.", log_file)

    # Example2: User request - Visualize the pairs of variables with high correlation
    corr = df.select_dtypes(include=[np.number]).corr()
    high_corr = corr.abs().unstack().sort_values(ascending=False)
    high_corr = high_corr[high_corr < 1].drop_duplicates()
    top_pairs = high_corr.head(3).index
    for var1, var2 in top_pairs:
        plt.figure()
        sns.scatterplot(x=df[var1], y=df[var2])
        plt.title(f'Scatter: {var1} vs {var2}')
        plt.savefig(f'{output_dir}/scatter_{var1}_vs_{var2}.png')
        plt.close()
        log_message(f"Plotted scatter for {var1} vs {var2}.", log_file)
    # === LLM_EDA_CUSTOM_ANALYSIS_END ===
    # (LLM: Remove or replace the above block as needed)
    """
    # Example:
    if 'usertype' in df.columns and 'tripduration' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='usertype', y='tripduration', data=df)
        plt.title('Trip Duration by User Type')
        plt.savefig(f'{output_dir}/tripduration_by_usertype.png')
        plt.close()
        log_message("Generated boxplot for tripduration by usertype.", log_file)

    if 'start station name' in df.columns:
        top_stations = df['start station name'].value_counts().head(10)
        top_stations.plot(kind='barh')
        plt.title('Top 10 Start Stations')
        plt.xlabel('Ride Count')
        plt.savefig(f'{output_dir}/top_start_stations.png')
        plt.close()
        log_message("Plotted top 10 start stations.", log_file)


def main():
    # === LLM_MAIN_RUN_START ===
    # (LLM will generate input file path and session ID)
    # IMPORTANT: {input_file} must be replaced with the actual cleaned file path from the inspection/cleaning step.
    session_id = '{session_id}'
    input_file = '{input_file}'  # (LLM: Replace with the cleaned file path, e.g., ic_full_output/{session_id}/cleaned_final_{uploaded_file_name}.csv)
    # === LLM_MAIN_RUN_END ===

    output_dir = create_output_directory(session_id)
    log_file_path = os.path.join(output_dir, 'eda_mixed_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_message("Starting mixed EDA pipeline.", log_file)

        # Load the cleaned dataset from the inspection/cleaning step
        df = pd.read_csv(input_file)
        log_message("Dataset loaded.", log_file)

        num_cols, cat_cols = analyze_dataset_structure(df, output_dir, log_file)
        distribution_analysis(df, num_cols, output_dir, log_file)
        correlation_and_vif(df, num_cols, output_dir, log_file)
        custom_analysis(df, output_dir, log_file)

        log_message("\n[LLM_EDA_SUMMARY_PLACEHOLDER]\nHere is the LLM-generated EDA interpretation/summary.", log_file)
        log_message("EDA mixed pipeline completed. Outputs saved.", log_file)

    print("EDA process finished. Check outputs folder.")
    # === LLM_EDA_SUMMARY_PRINT_START ===
    print("""[LLM_EDA_SUMMARY_PLACEHOLDER]\nHere is the text of the LLM-generated EDA interpretation/summary.\n""")
    # === LLM_EDA_SUMMARY_PRINT_END ===

if __name__ == "__main__":
    main()