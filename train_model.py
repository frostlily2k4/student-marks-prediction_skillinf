
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
data = pd.read_csv("student_data.csv")

print("STUDENT MARKS PREDICTION")
print("=" * 40)
print("\nDataset preview:")
print(data.head())

# 2. Separate features and target
X = data[["study_hours", "attendance", "previous_marks"]]
y = data["final_marks"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict marks on test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")

# 7. Display actual vs predicted marks
results = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": np.round(y_pred, 2)
})

print("\nTEST DATA PREDICTIONS")
print("-" * 40)
print(results.to_string(index=False))

# 8. Predict marks for a sample student
student = pd.DataFrame({
    "study_hours": [5],
    "attendance": [85],
    "previous_marks": [65]
})

predicted_marks = model.predict(student)[0]
predicted_marks = np.clip(predicted_marks, 0, 100)

print("\nSAMPLE STUDENT PREDICTION")
print("-" * 40)
print("Study hours per day: 5")
print("Attendance: 85%")
print("Previous marks: 65")
print(f"Estimated final marks: {predicted_marks:.2f}/100")

print("\nNote: This is an estimate, not a guarantee of actual marks.")
