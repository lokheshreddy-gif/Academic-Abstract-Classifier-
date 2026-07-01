"""
Visualization Script for Academic Abstract Classification Project
Generates: Outliers, Histograms, Univariate, Bivariate, Correlation, and Pair Plot Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Color palette
colors = ['#00d4ff', '#5b86e5', '#ff6b9d', '#ffa500', '#764ba2']

print("Loading datasets...")
# Load datasets
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# Combine for full analysis
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Create numerical features from text
full_df['abstract_length'] = full_df['abstract'].str.len()
full_df['word_count'] = full_df['abstract'].str.split().str.len()
full_df['sentence_count'] = full_df['abstract'].str.count(r'[.!?]+')
full_df['avg_word_length'] = full_df['abstract'].str.split().apply(
    lambda x: np.mean([len(word) for word in x]) if x else 0
)

# Label encoding for numerical analysis
label_map = {label: idx for idx, label in enumerate(sorted(full_df['label'].unique()))}
full_df['label_encoded'] = full_df['label'].map(label_map)

print("Generating visualizations...")

# ============================================================================
# 1. OUTLIER ANALYSIS
# ============================================================================
print("1. Generating Outlier Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Outlier Analysis - Academic Abstract Features', fontsize=16, fontweight='bold', y=0.995)

features = ['abstract_length', 'word_count', 'sentence_count', 'avg_word_length']
feature_names = ['Abstract Length (characters)', 'Word Count', 'Sentence Count', 'Average Word Length']

for idx, (feature, name) in enumerate(zip(features, feature_names)):
    ax = axes[idx // 2, idx % 2]
    
    # Box plot
    bp = ax.boxplot(full_df[feature].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor=colors[idx % len(colors)], alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    ax.set_title(f'{name} - Outlier Detection', fontweight='bold', pad=15)
    ax.set_ylabel('Value', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    Q1 = full_df[feature].quantile(0.25)
    Q3 = full_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = full_df[(full_df[feature] < lower_bound) | (full_df[feature] > upper_bound)]
    
    ax.text(0.5, 0.95, f'Outliers: {len(outliers)} ({len(outliers)/len(full_df)*100:.1f}%)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontweight='bold')

plt.tight_layout()
plt.savefig('1_outlier_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 1_outlier_analysis.png")
plt.close()

# ============================================================================
# 2. HISTOGRAM ANALYSIS (Visual Analysis)
# ============================================================================
print("2. Generating Histogram Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Histogram Analysis - Distribution of Features', fontsize=16, fontweight='bold', y=0.995)

for idx, (feature, name) in enumerate(zip(features, feature_names)):
    ax = axes[idx // 2, idx % 2]
    
    # Histogram with KDE
    n, bins, patches = ax.hist(full_df[feature].dropna(), bins=30, color=colors[idx % len(colors)],
                               alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add KDE curve
    from scipy.stats import gaussian_kde
    data = full_df[feature].dropna()
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range) * len(data) * (bins[1] - bins[0]), 
            'r-', linewidth=2, label='KDE')
    
    ax.set_title(f'{name} Distribution', fontweight='bold', pad=15)
    ax.set_xlabel(name, fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_val = full_df[feature].mean()
    median_val = full_df[feature].median()
    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('2_histogram_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 2_histogram_analysis.png")
plt.close()

# ============================================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================================
print("3. Generating Univariate Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Univariate Analysis - Individual Feature Distributions', fontsize=16, fontweight='bold', y=0.995)

# Feature 1: Abstract Length by Label
ax = axes[0, 0]
for label in sorted(full_df['label'].unique()):
    data = full_df[full_df['label'] == label]['abstract_length']
    ax.hist(data, alpha=0.6, label=label, bins=25, edgecolor='black')
ax.set_title('Abstract Length Distribution by Label', fontweight='bold')
ax.set_xlabel('Abstract Length (characters)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Feature 2: Word Count by Label
ax = axes[0, 1]
for label in sorted(full_df['label'].unique()):
    data = full_df[full_df['label'] == label]['word_count']
    ax.hist(data, alpha=0.6, label=label, bins=25, edgecolor='black')
ax.set_title('Word Count Distribution by Label', fontweight='bold')
ax.set_xlabel('Word Count', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Feature 3: Label Distribution
ax = axes[0, 2]
label_counts = full_df['label'].value_counts()
bars = ax.bar(range(len(label_counts)), label_counts.values, 
              color=colors[:len(label_counts)], edgecolor='black', linewidth=1.5)
ax.set_title('Label Distribution', fontweight='bold')
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_xticks(range(len(label_counts)))
ax.set_xticklabels(label_counts.index, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Feature 4: Sentence Count Distribution
ax = axes[1, 0]
for label in sorted(full_df['label'].unique()):
    data = full_df[full_df['label'] == label]['sentence_count']
    ax.hist(data, alpha=0.6, label=label, bins=20, edgecolor='black')
ax.set_title('Sentence Count Distribution by Label', fontweight='bold')
ax.set_xlabel('Sentence Count', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Feature 5: Average Word Length Distribution
ax = axes[1, 1]
for label in sorted(full_df['label'].unique()):
    data = full_df[full_df['label'] == label]['avg_word_length']
    ax.hist(data, alpha=0.6, label=label, bins=25, edgecolor='black')
ax.set_title('Average Word Length Distribution by Label', fontweight='bold')
ax.set_xlabel('Average Word Length', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Feature 6: Summary Statistics Box Plot
ax = axes[1, 2]
data_to_plot = [full_df[full_df['label'] == label]['abstract_length'].values 
                for label in sorted(full_df['label'].unique())]
bp = ax.boxplot(data_to_plot, labels=sorted(full_df['label'].unique()),
                patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Abstract Length by Label (Box Plot)', fontweight='bold')
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Abstract Length', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('3_univariate_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 3_univariate_analysis.png")
plt.close()

# ============================================================================
# 4. BIVARIATE ANALYSIS
# ============================================================================
print("4. Generating Bivariate Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bivariate Analysis - Feature Relationships', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Abstract Length vs Word Count
ax = axes[0, 0]
for label in sorted(full_df['label'].unique()):
    subset = full_df[full_df['label'] == label]
    ax.scatter(subset['abstract_length'], subset['word_count'], 
              alpha=0.6, label=label, s=50, edgecolors='black', linewidth=0.5)
ax.set_title('Abstract Length vs Word Count', fontweight='bold')
ax.set_xlabel('Abstract Length (characters)', fontweight='bold')
ax.set_ylabel('Word Count', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Word Count vs Sentence Count
ax = axes[0, 1]
for label in sorted(full_df['label'].unique()):
    subset = full_df[full_df['label'] == label]
    ax.scatter(subset['word_count'], subset['sentence_count'], 
              alpha=0.6, label=label, s=50, edgecolors='black', linewidth=0.5)
ax.set_title('Word Count vs Sentence Count', fontweight='bold')
ax.set_xlabel('Word Count', fontweight='bold')
ax.set_ylabel('Sentence Count', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Abstract Length vs Average Word Length by Label
ax = axes[1, 0]
for label in sorted(full_df['label'].unique()):
    subset = full_df[full_df['label'] == label]
    ax.scatter(subset['abstract_length'], subset['avg_word_length'], 
              alpha=0.6, label=label, s=50, edgecolors='black', linewidth=0.5)
ax.set_title('Abstract Length vs Average Word Length', fontweight='bold')
ax.set_xlabel('Abstract Length (characters)', fontweight='bold')
ax.set_ylabel('Average Word Length', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Violin Plot - Word Count by Label
ax = axes[1, 1]
data_to_plot = [full_df[full_df['label'] == label]['word_count'].values 
                for label in sorted(full_df['label'].unique())]
parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                      showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], colors[:len(parts['bodies'])]):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
ax.set_title('Word Count Distribution by Label (Violin Plot)', fontweight='bold')
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Word Count', fontweight='bold')
ax.set_xticks(range(len(sorted(full_df['label'].unique()))))
ax.set_xticklabels(sorted(full_df['label'].unique()), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('4_bivariate_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 4_bivariate_analysis.png")
plt.close()

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("5. Generating Correlation Analysis...")

# Select numerical features for correlation
numerical_features = ['abstract_length', 'word_count', 'sentence_count', 
                     'avg_word_length', 'label_encoded']
corr_df = full_df[numerical_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_df, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=2, 
            vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
ax.set_title('Correlation Matrix - Feature Relationships', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('5_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 5_correlation_analysis.png")
plt.close()

# ============================================================================
# 6. PAIR PLOT ANALYSIS
# ============================================================================
print("6. Generating Pair Plot Analysis...")

# Select key features for pair plot
pair_features = ['abstract_length', 'word_count', 'sentence_count', 'avg_word_length']
pair_df = full_df[pair_features + ['label']].dropna()

# Create pair plot
g = sns.pairplot(pair_df, hue='label', diag_kind='kde', 
                 palette=colors[:len(full_df['label'].unique())],
                 plot_kws={'alpha': 0.6, 's': 30, 'edgecolors': 'black', 'linewidth': 0.5},
                 diag_kws={'alpha': 0.7, 'linewidth': 2})
g.fig.suptitle('Pair Plot Analysis - Feature Relationships by Label', 
               fontsize=14, fontweight='bold', y=1.02)
g.fig.set_size_inches(14, 14)

# Adjust layout
plt.tight_layout()
plt.savefig('6_pair_plot_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 6_pair_plot_analysis.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Samples: {len(full_df)}")
print(f"Training Samples: {len(train_df)}")
print(f"Validation Samples: {len(val_df)}")
print(f"Test Samples: {len(test_df)}")
print(f"\nNumber of Labels: {len(full_df['label'].unique())}")
print(f"Labels: {', '.join(sorted(full_df['label'].unique()))}")

print("\n" + "-"*60)
print("FEATURE STATISTICS")
print("-"*60)
stats_df = full_df[features].describe()
print(stats_df.round(2))

print("\n" + "-"*60)
print("LABEL DISTRIBUTION")
print("-"*60)
print(full_df['label'].value_counts().sort_index())

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
print("\nGenerated Files:")
print("  1. 1_outlier_analysis.png")
print("  2. 2_histogram_analysis.png")
print("  3. 3_univariate_analysis.png")
print("  4. 4_bivariate_analysis.png")
print("  5. 5_correlation_analysis.png")
print("  6. 6_pair_plot_analysis.png")

