# 📦 Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📥 Load Dataset
df = pd.read_csv("Advertising.csv")

# 🧹 Clean Data (if needed)
df = df.drop(columns=["Unnamed: 0"], errors='ignore')  # Drop unnamed index column

# 🔍 EDA: Plotting
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.8)
plt.suptitle("Ad Spend vs Sales")
plt.show()

# 🎯 Define Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔎 Predict and Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("✅ R² Score:", r2)
print("📉 Mean Squared Error:", mse)

# 📈 Plot Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
