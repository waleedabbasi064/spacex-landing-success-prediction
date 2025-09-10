# src/evaluate.py

import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed test data
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_test.csv")
test_data = pd.read_csv(data_path)

X_test = test_data.drop("Class", axis=1)
y_test = test_data["Class"]

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "spacex_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Predictions
y_pred = model.predict(X_test)

# Print accuracy & classification report
print("✅ Model Evaluation Complete")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Success"], yticklabels=["Fail", "Success"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy Bar Plot
acc = accuracy_score(y_test, y_pred)
plt.figure(figsize=(4, 5))
plt.bar(["Accuracy"], [acc], color="green")
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.show()
