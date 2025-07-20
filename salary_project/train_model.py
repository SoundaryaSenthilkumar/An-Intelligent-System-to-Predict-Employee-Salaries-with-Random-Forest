# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data.csv")

# Make sure these column names match your actual dataset
X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"]  # Ensure this column exists and is 0/1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_model.pkl")

print("✅ Model trained and saved as rf_model.pkl")

