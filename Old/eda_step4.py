"""
EDA Step 4: Visual Summarization and Group-wise Analysis
This script creates visual summaries by:
1. Loading the processed dataset from step 3
2. Creating customer segments based on behavior
3. Generating group-wise visualizations
4. Summarizing key patterns and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path('eda_step4_output')
output_dir.mkdir(exist_ok=True)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the processed dataset from step 3"""
    return pd.read_csv('eda_step3_output/customer_data_processed.csv')

def create_customer_segments(df):
    """
    Create customer segments based on:
    1. Purchase Behavior (High/Medium/Low PctSessionPurchase)
    2. Engagement Level (High/Medium/Low SessionCount)
    3. Price Sensitivity (High/Medium/Low AvgPriceClicked)
    """
    segments = df.copy()
    
    # Create purchase behavior segments
    try:
        segments['purchase_segment'] = pd.qcut(
            df['PctSessionPurchase'],
            q=3,
            labels=['Low Purchase Rate', 'Medium Purchase Rate', 'High Purchase Rate'],
            duplicates='drop'
        )
    except ValueError:
        # If too many duplicates, use custom thresholds
        purchase_thresholds = df['PctSessionPurchase'].quantile([0.33, 0.67])
        segments['purchase_segment'] = pd.cut(
            df['PctSessionPurchase'],
            bins=[-np.inf, purchase_thresholds[0.33], purchase_thresholds[0.67], np.inf],
            labels=['Low Purchase Rate', 'Medium Purchase Rate', 'High Purchase Rate']
        )
    
    # Create engagement level segments
    try:
        segments['engagement_segment'] = pd.qcut(
            df['SessionCount'],
            q=3,
            labels=['Low Engagement', 'Medium Engagement', 'High Engagement'],
            duplicates='drop'
        )
    except ValueError:
        # If too many duplicates, use custom thresholds
        engagement_thresholds = df['SessionCount'].quantile([0.33, 0.67])
        segments['engagement_segment'] = pd.cut(
            df['SessionCount'],
            bins=[-np.inf, engagement_thresholds[0.33], engagement_thresholds[0.67], np.inf],
            labels=['Low Engagement', 'Medium Engagement', 'High Engagement']
        )
    
    # Create price sensitivity segments
    try:
        segments['price_segment'] = pd.qcut(
            df['AvgPriceClicked'],
            q=3,
            labels=['Low Price Range', 'Medium Price Range', 'High Price Range'],
            duplicates='drop'
        )
    except ValueError:
        # If too many duplicates, use custom thresholds
        price_thresholds = df['AvgPriceClicked'].quantile([0.33, 0.67])
        segments['price_segment'] = pd.cut(
            df['AvgPriceClicked'],
            bins=[-np.inf, price_thresholds[0.33], price_thresholds[0.67], np.inf],
            labels=['Low Price Range', 'Medium Price Range', 'High Price Range']
        )
    
    return segments

def plot_segment_distributions(segments):
    """Plot distribution of customers across different segments"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot purchase behavior distribution
    purchase_dist = segments['purchase_segment'].value_counts()
    axes[0].bar(purchase_dist.index, purchase_dist.values)
    axes[0].set_title('Distribution of Purchase Behavior')
    axes[0].set_ylabel('Number of Customers')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot engagement level distribution
    engagement_dist = segments['engagement_segment'].value_counts()
    axes[1].bar(engagement_dist.index, engagement_dist.values)
    axes[1].set_title('Distribution of Engagement Levels')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot price sensitivity distribution
    price_dist = segments['price_segment'].value_counts()
    axes[2].bar(price_dist.index, price_dist.values)
    axes[2].set_title('Distribution of Price Sensitivity')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'segment_distributions.png')
    plt.close()

def plot_segment_metrics(segments):
    """Plot key metrics across different segments"""
    metrics = ['AvgTimePerSession', 'ItemsClickedPerSession', 'PctSessionClickDiscount']
    segment_types = ['purchase_segment', 'engagement_segment', 'price_segment']
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, segment in enumerate(segment_types):
            sns.boxplot(data=segments, x=segment, y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} by {segment.split("_")[0].title()} Segment')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_by_segments.png')
        plt.close()

def plot_interaction_patterns(segments):
    """Plot interaction patterns between different segments"""
    # Create cross-tabulations
    purchase_vs_engagement = pd.crosstab(
        segments['purchase_segment'],
        segments['engagement_segment']
    )
    
    purchase_vs_price = pd.crosstab(
        segments['purchase_segment'],
        segments['price_segment']
    )
    
    engagement_vs_price = pd.crosstab(
        segments['engagement_segment'],
        segments['price_segment']
    )
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.heatmap(purchase_vs_engagement, annot=True, fmt='d', ax=axes[0])
    axes[0].set_title('Purchase Behavior vs Engagement Level')
    
    sns.heatmap(purchase_vs_price, annot=True, fmt='d', ax=axes[1])
    axes[1].set_title('Purchase Behavior vs Price Sensitivity')
    
    sns.heatmap(engagement_vs_price, annot=True, fmt='d', ax=axes[2])
    axes[2].set_title('Engagement Level vs Price Sensitivity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'segment_interactions.png')
    plt.close()

def analyze_time_patterns(segments):
    """Analyze and visualize patterns in time-related metrics"""
    time_metrics = ['AvgTimePerClick', 'AvgTimePerSession']
    
    for metric in time_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Time patterns by purchase behavior
        sns.violinplot(data=segments, x='purchase_segment', y=metric, ax=axes[0])
        axes[0].set_title(f'{metric} by Purchase Behavior')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Time patterns by engagement level
        sns.violinplot(data=segments, x='engagement_segment', y=metric, ax=axes[1])
        axes[1].set_title(f'{metric} by Engagement Level')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Time patterns by price sensitivity
        sns.violinplot(data=segments, x='price_segment', y=metric, ax=axes[2])
        axes[2].set_title(f'{metric} by Price Sensitivity')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_patterns.png')
        plt.close()

def generate_summary_report(segments):
    """Generate a summary report of the visual analysis"""
    with open(output_dir / 'visual_analysis_summary.txt', 'w') as f:
        f.write("Visual Analysis Summary\n")
        f.write("======================\n\n")
        
        # Segment sizes
        f.write("1. Segment Sizes:\n")
        for segment_type in ['purchase_segment', 'engagement_segment', 'price_segment']:
            f.write(f"\n{segment_type.split('_')[0].title()} Segments:\n")
            segment_counts = segments[segment_type].value_counts()
            for segment, count in segment_counts.items():
                percentage = (count / len(segments)) * 100
                f.write(f"- {segment}: {count} customers ({percentage:.1f}%)\n")
        
        # Key metrics by segment
        f.write("\n2. Key Metrics by Segment:\n")
        metrics = ['AvgTimePerSession', 'ItemsClickedPerSession', 'PctSessionClickDiscount']
        for segment_type in ['purchase_segment', 'engagement_segment', 'price_segment']:
            f.write(f"\n{segment_type.split('_')[0].title()} Segment Metrics:\n")
            for metric in metrics:
                f.write(f"\n{metric}:\n")
                agg_stats = segments.groupby(segment_type)[metric].agg(['mean', 'std'])
                for segment in agg_stats.index:
                    f.write(f"- {segment}: {agg_stats.loc[segment, 'mean']:.2f} "
                           f"(±{agg_stats.loc[segment, 'std']:.2f})\n")
        
        # Segment interactions
        f.write("\n3. Segment Interactions:\n")
        # Purchase vs Engagement
        purchase_engagement = pd.crosstab(
            segments['purchase_segment'],
            segments['engagement_segment'],
            normalize='index'
        ) * 100
        f.write("\nPurchase Behavior vs Engagement Level (row percentages):\n")
        f.write(purchase_engagement.round(1).to_string())
        
        # Purchase vs Price
        purchase_price = pd.crosstab(
            segments['purchase_segment'],
            segments['price_segment'],
            normalize='index'
        ) * 100
        f.write("\n\nPurchase Behavior vs Price Sensitivity (row percentages):\n")
        f.write(purchase_price.round(1).to_string())

def main():
    # Load data
    df = load_data()
    
    # Create customer segments
    segments = create_customer_segments(df)
    
    # Generate visualizations
    plot_segment_distributions(segments)
    plot_segment_metrics(segments)
    plot_interaction_patterns(segments)
    analyze_time_patterns(segments)
    
    # Generate summary report
    generate_summary_report(segments)
    
    # Save segmented data
    segments.to_csv(output_dir / 'customer_segments.csv', index=False)

if __name__ == "__main__":
    main() 
