import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# === LLM_INITIAL_SETUP_START ===
# (LLM will generate code to set up initial parameters and configurations)
# Example:
# session_id = '{session_id}'
# input_file = '{input_file}'
# output_base_dir = 'ic_full_output'
# === LLM_INITIAL_SETUP_END ===

def create_output_directory(session_id):
    """Create a single output directory"""
    base_dir = f'ic_full_output/{session_id}'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def step1_data_cleaning(df, output_dir, summary_logs):
    """Step 1: Basic data cleaning and validation
    LLM will generate code to:
    1. Analyze data types and formats
    2. Clean whitespace and capitalization
    3. Generate summary statistics
    4. Save cleaned data and logs
    
    The following is example code that can be modified by LLM based on the specific dataset.
    """
    print("Starting Step 1: Basic data cleaning and validation...")
    log_path = f'{output_dir}/step1_data_cleaning_log.txt'

    # === LLM_STEP1_CODE_START ===
    with open(log_path, 'w') as log_file:
        log_file.write("Data Cleaning Log - Step 1\n" + "=" * 50 + "\n\n")
        log_file.write(f"Total rows: {len(df)}\n")
        log_file.write(f"Total columns: {len(df.columns)}\n\n")

        column_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        log_file.write(column_info.to_string() + "\n\n")

        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if df[col].str.strip().ne(df[col]).any():
                log_file.write(f"Found whitespace issues in column: {col}\n")
                df[col] = df[col].str.strip()

        for col in string_columns:
            if df[col].nunique() < 50:
                unique_values = df[col].unique()
                if any(str(val).islower() for val in unique_values if pd.notnull(val)) and \
                   any(str(val).isupper() for val in unique_values if pd.notnull(val)):
                    log_file.write(f"Inconsistent capitalization found in column: {col}\n")

        log_file.write("\nSummary Statistics:\n" + "-" * 30 + "\n")
        log_file.write(df.describe().to_string())
    # === LLM_STEP1_CODE_END ===

    summary_logs.append(f"Step 1 completed. See {log_path}")
    return df

def step2_duplicate_handling(df, output_dir, summary_logs):
    """Step 2: Handle duplicate records
    LLM will generate code to:
    1. Detect exact duplicates
    2. Identify near-duplicates
    3. Handle duplicates based on business rules
    4. Save cleaned data and logs
    
    The following is example code that can be modified by LLM based on the specific dataset.
    """
    print("Starting Step 2: Duplicate handling...")
    log_path = f'{output_dir}/step2_duplicate_handling_log.txt'

    # === LLM_STEP2_CODE_START ===
    with open(log_path, 'w') as log_file:
        log_file.write("Duplicate Handling Log - Step 2\n" + "=" * 50 + "\n\n")
        initial_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        log_file.write(f"Initial rows: {initial_rows}\n")
        log_file.write(f"Exact duplicates: {duplicate_rows}\n")

        df_no_exact_dupes = df.drop_duplicates()
        log_file.write(f"Rows after removing exact duplicates: {len(df_no_exact_dupes)}\n")

        key_columns = ['Start Time', 'Stop Time', 'Start Station ID', 'End Station ID', 'Bike ID']
        near_dupes = df_no_exact_dupes.duplicated(subset=key_columns, keep=False)
        near_dupe_count = near_dupes.sum()
        log_file.write(f"Potential near-duplicates: {near_dupe_count}\n")

        if near_dupe_count > 0:
            df_cleaned = df_no_exact_dupes.drop_duplicates(subset=key_columns, keep='first')
        else:
            df_cleaned = df_no_exact_dupes
    # === LLM_STEP2_CODE_END ===

    summary_logs.append(f"Step 2 completed. See {log_path}")
    return df_cleaned

def step3_missing_values(df, output_dir, summary_logs):
    """Step 3: Handle missing values
    LLM will generate code to:
    1. Analyze missing data patterns
    2. Apply appropriate imputation methods
    3. Create imputation flags
    4. Generate visualizations
    5. Save cleaned data and logs
    
    The following is example code that can be modified by LLM based on the specific dataset.
    """
    print("Starting Step 3: Missing values handling...")
    log_path = f'{output_dir}/step3_missing_values_log.txt'

    # === LLM_STEP3_CODE_START ===
    with open(log_path, 'w') as log_file:
        log_file.write("Missing Values Analysis - Step 3\n" + "=" * 50 + "\n\n")
        missing_stats = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        log_file.write(missing_stats.to_string() + "\n\n")

        df_cleaned = df.copy()
        imputed_columns = []

        for column in df.columns:
            missing_pct = (df[column].isnull().sum() / len(df)) * 100
            if 5 <= missing_pct <= 50:
                if df[column].dtype == 'object':
                    mode_value = df[column].mode()[0]
                    df_cleaned[column] = df[column].fillna(mode_value)
                    log_file.write(f"{column} imputed with mode: {mode_value}\n")
                else:
                    median_value = df[column].median()
                    df_cleaned[column] = df[column].fillna(median_value)
                    log_file.write(f"{column} imputed with median: {median_value}\n")
                imputed_columns.append(column)

        for column in imputed_columns:
            flag_col = f"{column}_imputed"
            df_cleaned[flag_col] = df[column].isnull().astype(int)
            log_file.write(f"Flag column created: {flag_col}\n")

        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.savefig(f'{output_dir}/missing_values_heatmap.png')
        plt.close()
    # === LLM_STEP3_CODE_END ===

    summary_logs.append(f"Step 3 completed. See {log_path}")
    return df_cleaned

def step4_outlier_detection(df, output_dir, summary_logs):
    """Step 4: Detect and handle outliers
    LLM will generate code to:
    1. Detect outliers using multiple methods
    2. Apply appropriate outlier treatment
    3. Create outlier flags
    4. Generate visualizations
    5. Save cleaned data and logs
    
    The following is example code that can be modified by LLM based on the specific dataset.
    """
    print("Starting Step 4: Outlier detection...")
    log_path = f'{output_dir}/step4_outlier_detection_log.txt'

    # === LLM_STEP4_CODE_START ===
    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

    def detect_outliers_zscore(series):
        return np.abs(stats.zscore(series)) > 3

    df_cleaned = df.copy()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    with open(log_path, 'w') as log_file:
        log_file.write("Outlier Detection - Step 4\n" + "=" * 50 + "\n\n")
        for column in numerical_columns:
            iqr_outliers = detect_outliers_iqr(df[column])
            zscore_outliers = detect_outliers_zscore(df[column])
            iqr_count = iqr_outliers.sum()
            zscore_count = zscore_outliers.sum()
            method = "IQR" if iqr_count < zscore_count else "Z-score"
            outliers = iqr_outliers if method == "IQR" else zscore_outliers

            if outliers.any():
                median_value = df[column].median()
                df_cleaned.loc[outliers, column] = median_value
                log_file.write(f"{column}: {outliers.sum()} outliers replaced with median using {method}\n")

            plt.figure(figsize=(10, 6))
            plt.boxplot([df[column], df_cleaned[column]], labels=['Original', 'Cleaned'])
            plt.title(f'Boxplot for {column}')
            plt.savefig(f'{output_dir}/{column}_boxplot.png')
            plt.close()
    # === LLM_STEP4_CODE_END ===

    summary_logs.append(f"Step 4 completed. See {log_path}")
    return df_cleaned

def main():
    """Main function to run the inspection and cleaning pipeline
    LLM will generate code to:
    1. Load input data
    2. Execute all steps
    3. Save final cleaned data
    4. Generate summary report
    
    The following is example code that can be modified by LLM based on the specific dataset.
    """
    # === LLM_MAIN_EXECUTION_START ===
    # (LLM: Replace with actual session_id, input_file, uploaded_file_name)
    session_id = '{session_id}'
    input_file = '{input_file}'
    uploaded_file_name = '{uploaded_file_name}'

    output_dir = create_output_directory(session_id)

    print("Reading the dataset...")
    df = pd.read_csv(input_file)

    summary_logs = []

    df = step1_data_cleaning(df, output_dir, summary_logs)
    df = step2_duplicate_handling(df, output_dir, summary_logs)
    df = step3_missing_values(df, output_dir, summary_logs)
    df = step4_outlier_detection(df, output_dir, summary_logs)

    # Save final cleaned data
    cleaned_path = f'{output_dir}/cleaned_final_{uploaded_file_name}.csv'
    df.to_csv(cleaned_path, index=False)
    print("Final cleaned data saved.")

    # Save summary log file
    summary_log_path = f'{output_dir}/log_ic.txt'
    with open(summary_log_path, 'w') as log_file:
        log_file.write("Summary Log for Inspection & Cleaning\n" + "=" * 50 + "\n\n")
        for entry in summary_logs:
            log_file.write(entry + "\n")
    print("Summary log file saved.")
    # === LLM_MAIN_EXECUTION_END ===

if __name__ == "__main__":
    main()