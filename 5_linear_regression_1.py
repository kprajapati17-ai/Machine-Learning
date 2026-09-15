import numpy as np
from sklearn.linear_model import LinearRegression

# Input feature:
# Daily study hours
study_hours = np.array([
    [1.0],
    [1.5],
    [2.0],
    [2.5],
    [3.0],
    [3.5],
    [4.0],
    [4.5],
    [5.0],
    [5.5],
    [6.0],
    [6.5]
])

# Target:
# Student marks
marks = np.array([
    30, 35, 40, 45, 49, 52,
    55, 60, 65, 70, 75, 80
])


model = LinearRegression()
model.fit(study_hours,marks)
print("Model trained successfully.")

hours = float(input("Enter person's daily study hours: "))
person = np.array([[hours]])

prediction = model.predict(person)
predicted_marks = prediction[0]

print(f"Predicted marks for {hours} study hours: {predicted_marks:.2f}")

print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept:{model.intercept_:.2f}")
