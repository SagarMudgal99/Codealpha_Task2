import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load dataset from previous scraping task
df = pd.read_csv('books_custom.csv')
print(f"📊 Loaded {df.shape[0]} books from {df.shape[1]} columns")

# ● TASK 2.1: ASK MEANINGFUL QUESTIONS
# ===============================================
print("\n" + "="*60)
print("🔍 MEANINGFUL BUSINESS QUESTIONS")
print("="*60)
questions = {
    "Q1": "Do higher-rated books cost more? (Price-Rating correlation)",
    "Q2": "Which ratings have stock shortages?",
    "Q3": "Are there pricing anomalies or outliers?",
    "Q4": "What's the price distribution shape?",
    "Q5": "Which price segments sell best by rating?"
}
for q_num, question in questions.items():
    print(f"{q_num}: {question}")

# ===============================================
# ● TASK 2.2: EXPLORE DATA STRUCTURE
# ===============================================
print("\n" + "="*60)
print("📋 DATA STRUCTURE EXPLORATION")
print("="*60)

print("\n1. DATA TYPES & STRUCTURE:")
print(df.dtypes.to_string())
print(f"\nDataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n2. BASIC STATISTICS:")
print(df.describe(include='all').round(2))

print("\n3. CATEGORICAL COUNTS:")
print("\nRating distribution:")
print(df['rating'].value_counts(normalize=True).round(3)*100, " %")
print("\nAvailability:")
print(df['availability'].value_counts())

# Data cleaning for analysis
df['price_numeric'] = df['price'].str.replace('£', '').astype(float)
df['in_stock'] = df['availability'].str.contains('In stock', case=False)

# ===============================================
# ● TASK 2.3: IDENTIFY TRENDS, PATTERNS, ANOMALIES
# ===============================================
print("\n" + "="*60)
print("📈 TRENDS, PATTERNS & ANOMALIES")
print("="*60)

# Create comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 15))

# 1. Price distribution & outliers
plt.subplot(3, 3, 1)
plt.hist(df['price_numeric'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price (£)')
plt.ylabel('Frequency')

# IQR outlier detection
Q1, Q3 = df['price_numeric'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df['price_numeric'] < Q1-1.5*IQR) | (df['price_numeric'] > Q3+1.5*IQR)]
print(f"\n🔴 PRICE OUTLIERS: {len(outliers)} books ({len(outliers)/len(df)*100:.1f}%)")
print(f"   Range: £{outliers['price_numeric'].min():.2f} - £{outliers['price_numeric'].max():.2f}")

plt.subplot(3, 3, 2)
rating_order = ['Five', 'Four', 'Three', 'Two', 'One']
sns.boxplot(data=df, x='rating', y='price_numeric', order=rating_order)
plt.title('Price by Rating (Trend Analysis)')
plt.xticks(rotation=45)

# 2. Rating vs Stock pattern
plt.subplot(3, 3, 3)
stock_table = pd.crosstab(df['rating'], df['in_stock'])
stock_table.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Stock Availability by Rating')
plt.legend(title='In Stock')

# 3. Price segments
plt.subplot(3, 3, 4)
df['price_segment'] = pd.cut(df['price_numeric'], bins=[0, 20, 40, 60, 100, np.inf], 
                            labels=['Budget', 'Mid', 'Premium', 'Luxury', 'Extreme'])
segment_rating = df.groupby('price_segment')['rating'].value_counts(normalize=True).unstack().fillna(0)
sns.heatmap(segment_rating, annot=True, cmap='YlGnBu', fmt='.1%')
plt.title('Rating Distribution by Price Segment')

plt.tight_layout()
plt.savefig('eda_complete_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ===============================================
# ● TASK 2.4: TEST HYPOTHESES & VALIDATE ASSUMPTIONS
# ===============================================
print("\n" + "="*60)
print("🧪 STATISTICAL HYPOTHESIS TESTING")
print("="*60)

print("\nHYPOTHESIS 1: Price distribution is normal")
stat, p_value = stats.shapiro(df['price_numeric'].sample(5000))
print(f"   Shapiro-Wilk test: p={p_value:.4f}")
print(f"   {'❌ REJECTED: Right-skewed prices' if p_value < 0.05 else '✅ Normal distribution'}")

print("\nHYPOTHESIS 2: No price difference across ratings (ANOVA)")
groups = [group['price_numeric'].values for name, group in df.groupby('rating')]
f_stat, f_p = stats.f_oneway(*groups)
print(f"   ANOVA F-test: p={f_p:.4f}")
print(f"   {'✅ Significant price differences by rating' if f_p < 0.05 else '❌ No difference'}")

print("\nHYPOTHESIS 3: Rating & stock availability independent")
chi2, chi_p, dof, expected = stats.chi2_contingency(pd.crosstab(df['rating'], df['in_stock']))
print(f"   Chi-square test: p={chi_p:.4f}")
print(f"   {'✅ Stock depends on rating' if chi_p < 0.05 else '❌ Independent'}")

# ===============================================
# ● TASK 2.5: DETECT DATA ISSUES & PROBLEMS
# ===============================================
print("\n" + "="*60)
print("🚨 DATA QUALITY & INTEGRITY REPORT")
print("="*60)

quality_checks = {}

# Data quality metrics
quality_checks['missing_values'] = df.isnull().sum().sum()
quality_checks['duplicates'] = df.duplicated().sum()
quality_checks['zero_prices'] = (df['price_numeric'] == 0).sum()
quality_checks['outliers'] = len(outliers)
quality_checks['stock_out'] = (~df['in_stock']).sum()

# Print quality report
print("\nDATA QUALITY SUMMARY:")
for metric, count in quality_checks.items():
    severity = "🔴 CRITICAL" if count > 50 else "🟡 WARNING" if count > 5 else "🟢 CLEAN"
    pct = count / len(df) * 100
    print(f"  {severity}: {metric.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

print("\nPRICE STATISTICS:")
print(f"  Mean: £{df['price_numeric'].mean():.2f}")
print(f"  Median: £{df['price_numeric'].median():.2f}")
print(f"  Skewness: {df['price_numeric'].skew():.2f}")
print(f"  Kurtosis: {stats.kurtosis(df['price_numeric']):.2f}")

# ===============================================
# RECOMMENDATIONS FOR FURTHER ANALYSIS
# ===============================================
print("\n" + "="*60)
print("🎯 ACTIONABLE INSIGHTS & RECOMMENDATIONS")
print("="*60)
print("📊 KEY FINDINGS:")
print(f"• Five-star books: {df['rating'].value_counts().get('Five', 0)} ({df['rating'].value_counts(normalize=True).get('Five', 0)*100:.1f}%)")
print(f"• Average price: £{df['price_numeric'].mean():.2f} ± £{df['price_numeric'].std():.2f}")
print(f"• Stock availability: {df['in_stock'].mean():.1%}")
print(f"• Price outliers need investigation: {len(outliers)} books")

print("\n🔧 NEXT STEPS:")
print("1. Log-transform prices for modeling (right-skewed)")
print("2. Remove/investigate {len(outliers)} price outliers")
print("3. Create price_rating interaction features")
print("4. Segment analysis by price tiers")
print("5. Time-series if stock data available")

print("\n✅ EDA COMPLETE | Visualizations saved: eda_complete_analysis.png")
