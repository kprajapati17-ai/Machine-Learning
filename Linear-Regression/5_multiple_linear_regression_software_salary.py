import numpy as np
from sklearn.linear_model import LinearRegression

# Column 1 = Years of Professional Coding Experience
# Column 2 = Number of Specialized Technical Certifications

X = np.array([
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],

    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 1],
    [7, 1],
    [8, 1],
    [9, 1],
    [10, 1],

    [2, 2],
    [3, 2],
    [4, 2],
    [5, 2],
    [6, 2],
    [7, 2],
    [8, 2],
    [9, 2],
    [10, 2],
    [11, 2],

    [3, 3],
    [4, 3],
    [5, 3],
    [6, 3],
    [7, 3],
    [8, 3],
    [9, 3],
    [10, 3],
    [11, 3],
    [12, 3]
])


# y = Annual Salary in $1,000s

y = np.array([
     55,  62,  69,  76,  83,
     90,  97, 104, 111, 118,

     67,  74,  81,  88,  95,
    102, 109, 116, 123, 130,

     79,  86,  93, 100, 107,
    114, 121, 128, 135, 142,

     91,  98, 105, 112, 119,
    126, 133, 140, 147, 154
])

model = LinearRegression()

model.fit(X, y)

print("\nModel trained successfully.")

experience = float(
    input("\nEnter years of coding experience: ")
)

certifications = float(
    input("Enter number of technical certifications: ")
)

developer = np.array([
    [experience, certifications]
])

predicted_salary = model.predict(developer)[0]

print("\n========== SALARY PREDICTION ==========")
print(f"Experience: {experience:.1f} years")
print(f"Certifications: {certifications:.0f}")
print(f"Predicted Annual Salary: ${predicted_salary:.2f}K")

