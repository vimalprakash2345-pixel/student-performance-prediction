import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv("student_data.csv")

print("Dataset shape:", data.shape)
print("\nFirst 5 records:")
print(data.head())

# Separate inputs and target
X = data[["study_hours", "attendance", "previous_marks"]]
y = data["final_score"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Compare actual scores with predicted scores
plt.scatter(y_test, y_pred)

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Scores")
plt.grid(True)
plt.show()


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel trained successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


# Get input from the user
study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))
previous_marks = float(input("Enter previous marks: "))

# Create input data for the model
new_student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_marks": [previous_marks]
})

# Predict the final score
predicted_score = model.predict(new_student)

print("\nPredicted Final Score:", round(predicted_score[0], 2))


import matplotlib.pyplot as plt

# Visualize study hours and final scores
plt.scatter(
    data["study_hours"],
    data["final_score"]
)

plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")

plt.grid(True)
plt.show()
