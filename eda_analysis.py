# ======================================================
# Real-world Data Project - Retail Sales Analysis
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (11, 6)

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = sns.load_dataset('tips')   # Restaurant sales data (Real-world retail example)

print("===== Dataset Overview =====")
print("Shape:", df.shape)
print("\nFirst 8 rows:")
print(df.head(8))
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# 2. Basic Statistics
# -------------------------------
print("\n===== Statistical Summary =====")
print(df.describe().round(2))

# -------------------------------
# 3. Feature Engineering
# -------------------------------
df['tip_percentage'] = (df['tip'] / df['total_bill']) * 100

print("\n===== Tip Percentage Summary =====")
print(df['tip_percentage'].describe().round(2))

# -------------------------------
# 4. Visualizations
# -------------------------------

# 1. Total Bill & Tip Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df['total_bill'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Total Bill Distribution')
sns.histplot(df['tip'], kde=True, ax=axes[1], color='orange')
axes[1].set_title('Tip Distribution')
plt.tight_layout()
plt.show()

# 2. Sales by Day
plt.figure(figsize=(9, 5))
sns.barplot(x='day', y='total_bill', data=df, estimator=np.sum, palette='viridis', ci=None)
plt.title('Total Sales by Day')
plt.ylabel('Total Sales ($)')
plt.show()

# 3. Average Tip % by Day & Time
plt.figure(figsize=(10, 5))
sns.barplot(x='day', y='tip_percentage', hue='time', data=df, palette='Set2', ci=None)
plt.title('Average Tip Percentage by Day & Time')
plt.ylabel('Tip Percentage (%)')
plt.show()

# 4. Total Bill vs Tip (with size)
plt.figure(figsize=(9, 6))
sns.scatterplot(x='total_bill', y='tip', hue='size', size='size',
                data=df, palette='coolwarm', sizes=(40, 200), alpha=0.7)
plt.title('Total Bill vs Tip (Bubble size = Party Size)')
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[['total_bill', 'tip', 'size', 'tip_percentage']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# -------------------------------
# 5. Key Business Insights
# -------------------------------
print("\n" + "="*55)
print("           KEY BUSINESS INSIGHTS")
print("="*55)

print(f"\n1. Total Records          : {df.shape[0]}")
print(f"2. Average Bill Amount    : ${df['total_bill'].mean():.2f}")
print(f"3. Average Tip Amount     : ${df['tip'].mean():.2f}")
print(f"4. Average Tip Percentage : {df['tip_percentage'].mean():.1f}%")
print(f"5. Highest Sales Day      : {df.groupby('day')['total_bill'].sum().idxmax()}")
print(f"6. Best Tip % Day         : {df.groupby('day')['tip_percentage'].mean().idxmax()}")

print("\n7. Dinner brings more revenue than Lunch.")
print("8. Larger party size → Higher total bill.")
print("9. Strong positive correlation between Total Bill and Tip.")

print("\n===== Project Completed Successfully! =====")