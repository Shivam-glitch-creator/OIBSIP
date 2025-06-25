# 📦 Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 📥 Load Dataset
df = pd.read_csv("car data.csv")

# 🧹 Data Preprocessing
df['Car_Age'] = 2025 - df['Year']  # Create age feature
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)  # Drop unnecessary columns

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# 🎯 Define Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# 🔀 Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train the Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 🔎 Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📈 Model Performance:")
print(f"Mean Absolute Error: {mae:.2f} lakhs")
print(f"R² Score: {r2:.2f}")

# 📊 Optional: Plot Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', figsize=(10, 6), color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
