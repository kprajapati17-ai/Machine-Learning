import numpy as np
from sklearn.linear_model import LinearRegression

# Input Features
# Column 1 = Hours Studied per Week
# Column 2 = Attendance Rate (%)
import numpy as np

# Input Features
# Column 1 = Hours Studied per Week
# Column 2 = Attendance Rate (%)

X = np.array([
    [5, 60],
    [8, 65],
    [10, 70],
    [12, 72],
    [15, 75],
    [18, 78],
    [20, 80],
    [22, 82],
    [25, 85],
    [28, 88],
    [30, 90],
    [32, 92],
    [35, 94],
    [38, 95],
    [40, 96],
    [42, 97],
    [45, 98],
    [48, 99],
    [50, 100],
    [52, 100]
])


# Target Variable
# y = Final Exam Score (out of 100)

y = np.array([
    45,
    50,
    55,
    58,
    62,
    66,
    68,
    70,
    74,
    78,
    80,
    82,
    85,
    87,
    89,
    91,
    93,
    95,
    97,
    98
])

print(X.shape)
print(y.shape)

model = LinearRegression()
model.fit(X,y)
print("Model trained successfully.")

hours = int(input("Enter Hours Studied per Week"))
attendce = int(input("Enter Attendance Rate (%)"))
student = np.array([[hours,attendce]])

pre = model.predict(student)[0]
print(f"student Hours Studied per Week: {student[0,0]:.2f}")
print(f"student Attendance Rate (%): {student[0,1]:.2f}%")
print(f"Final Exam Score (out of 100): {pre:.2f}")


