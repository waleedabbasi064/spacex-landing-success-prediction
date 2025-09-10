# src/preprocess.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load raw dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "spacex_launch_data.csv")
df = pd.read_csv(data_path)

# Encode categorical columns
label_encoders = {}
for col in ["Orbit", "LaunchSite"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits for later
train_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_train.csv")
test_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_test.csv")

train_data = X_train.copy()
train_data["Class"] = y_train
train_data.to_csv(train_path, index=False)

test_data = X_test.copy()
test_data["Class"] = y_test
test_data.to_csv(test_path, index=False)

print("✅ Data preprocessing complete")
print(f"Train saved at: {train_path}")
print(f"Test saved at: {test_path}")
