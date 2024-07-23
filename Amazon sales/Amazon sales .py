import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/svelo/Downloads/Amazon Sales data.csv"
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Show the first few rows of the dataset
print("\nFirst Few Rows of the Dataset:")
print(data.head())

# ETL - Extract
# Check for missing values
print("\nMissing Values in the Dataset:")
print(data.isnull().sum())

# Fill missing values if any
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data['Ship Date'] = pd.to_datetime(data['Ship Date'], errors='coerce')
data.fillna({'Order Priority': 'Not Specified'}, inplace=True)

# Transform
# Convert Order Date and Ship Date to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

# Create new columns for analysis
data['Order Month'] = data['Order Date'].dt.to_period('M').astype(str)  # Ensure Order Month is string
data['Order Year'] = data['Order Date'].dt.year
data['Order YearMonth'] = data['Order Date'].dt.to_period('M').astype(str)  # Ensure Order YearMonth is string

# Calculate total revenue, cost, and profit by month and year
monthly_sales = data.groupby('Order Month').agg({
    'Total Revenue': 'sum',
    'Total Cost': 'sum',
    'Total Profit': 'sum'
}).reset_index()

yearly_sales = data.groupby('Order Year').agg({
    'Total Revenue': 'sum',
    'Total Cost': 'sum',
    'Total Profit': 'sum'
}).reset_index()

yearly_monthly_sales = data.groupby('Order YearMonth').agg({
    'Total Revenue': 'sum',
    'Total Cost': 'sum',
    'Total Profit': 'sum'
}).reset_index()

# Key metrics
key_metrics = {
    'Total Revenue': data['Total Revenue'].sum(),
    'Total Cost': data['Total Cost'].sum(),
    'Total Profit': data['Total Profit'].sum(),
    'Average Profit Margin': data['Total Profit'].sum() / data['Total Revenue'].sum() * 100
}

print("\nKey Metrics:")
for key, value in key_metrics.items():
    print(f"{key}: {value:.2f}")

# Visualization for Sales Trend
# Monthly Sales Trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='Order Month', y='Total Revenue', data=monthly_sales, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Yearly Sales Trend
plt.figure(figsize=(12, 6))
sns.barplot(x='Order Year', y='Total Revenue', data=yearly_sales, palette='viridis')
plt.title('Yearly Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()

# Yearly and Monthly Sales Trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='Order YearMonth', y='Total Revenue', data=yearly_monthly_sales, marker='o')
plt.title('Yearly and Monthly Sales Trend')
plt.xlabel('Year-Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization for Relationships
# Sales Channel and Total Revenue
plt.figure(figsize=(12, 6))
sns.barplot(x='Sales Channel', y='Total Revenue', data=data, estimator='sum', palette='viridis')
plt.title('Total Revenue by Sales Channel')
plt.xlabel('Sales Channel')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Region and Total Profit
plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Total Profit', data=data, palette='viridis')
plt.title('Total Profit Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Item Type and Total Profit
plt.figure(figsize=(12, 6))
sns.barplot(x='Item Type', y='Total Profit', data=data, estimator='sum', palette='viridis')
plt.title('Total Profit by Item Type')
plt.xlabel('Item Type')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization for Key Metrics
# Key Metrics Bar Chart
plt.figure(figsize=(10, 6))
metrics_labels = list(key_metrics.keys())
metrics_values = list(key_metrics.values())
sns.barplot(x=metrics_labels, y=metrics_values, palette='viridis')
plt.title('Key Metrics')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# Profit Margin Pie Chart
plt.figure(figsize=(8, 8))
sizes = [key_metrics['Total Profit'], key_metrics['Total Revenue'] - key_metrics['Total Profit']]
labels = ['Total Profit', 'Remaining Revenue']
colors = ['#ff9999','#66b3ff']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Profit Margin Distribution')
plt.show()
