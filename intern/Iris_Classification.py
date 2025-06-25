# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
iris_df = pd.read_csv("Iris.csv")

# Step 2: Data preprocessing
iris_df = iris_df.drop(columns=["Id"])  # Remove unnecessary 'Id' column

# Encode the target column (Species) into numeric values
le = LabelEncoder()
iris_df["Species"] = le.fit_transform(iris_df["Species"])

# Step 3: Define features and target
X = iris_df.drop(columns=["Species"])  # Features: Sepal & Petal dimensions
y = iris_df["Species"]                 # Target: Encoded species label

# Step 4: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model using Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Predict on the test set
y_pred = clf.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

# Output the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
