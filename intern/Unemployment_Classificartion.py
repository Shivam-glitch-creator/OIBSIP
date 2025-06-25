# =====================
# 📦 1. Import Libraries
# =====================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# =====================
# 📥 2. Load Dataset
# =====================
df = pd.read_csv("Unemployment in India.csv")

# =====================
# 🔍 3. Inspect Dataset
# =====================
print("Dataset Info:\n")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nSample Data:\n", df.head())

# =============================
# 🧹 4. Data Preprocessing
# =============================
# Rename columns for easier access
df.columns = ['State', 'Date', 'Frequency', 'Estimated Unemployment Rate',
              'Estimated Employed', 'Estimated Labour Participation Rate', 'Area']

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# =============================
# 📊 5. Exploratory Data Analysis
# =============================

# --- a. Unemployment Rate Over Time ---
plt.figure(figsize=(14, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate', data=df, label='All Regions')
plt.title('Overall Unemployment Rate Over Time in India')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- b. State-wise Average Unemployment Rate ---
state_avg = df.groupby('State')['Estimated Unemployment Rate'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=state_avg.values, y=state_avg.index, palette="coolwarm")
plt.title("Average Unemployment Rate by State (2020–2021)")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# --- c. Rural vs Urban Comparison ---
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate', hue='Area')
plt.title('Rural vs Urban Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(title="Area")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- d. Monthly Trends ---
df['Month'] = df['Date'].dt.month_name()
monthly_avg = df.groupby('Month')['Estimated Unemployment Rate'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette="viridis")
plt.title("Average Monthly Unemployment Rate")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
