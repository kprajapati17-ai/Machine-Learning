import numpy as np
from sklearn.linear_model import LinearRegression

# Input Features
# Column 1 = Active Premium Members
# Column 2 = Monthly Digital Marketing Expenditure ($)

X = np.array([
    [100,  500],
    [150,  700],
    [200,  900],
    [250, 1100],
    [300, 1300],
    [350, 1500],
    [400, 1700],
    [450, 1900],
    [500, 2100],
    [550, 2300],

    [120,  600],
    [180,  800],
    [220, 1000],
    [280, 1200],
    [330, 1400],
    [380, 1600],
    [430, 1800],
    [480, 2000],
    [530, 2200],
    [580, 2400],

    [140,  650],
    [190,  850],
    [240, 1050],
    [290, 1250],
    [340, 1450],
    [390, 1650],
    [440, 1850],
    [490, 2050],
    [540, 2250],
    [590, 2450]
])

# Target Variable
# y = Total Monthly Revenue ($)

y = np.array([
     9000, 12000, 15000, 18000, 21000,
    24000, 27000, 30000, 33000, 36000,

    10200, 13800, 16200, 19800, 22800,
    25800, 28800, 31800, 34800, 37800,

    11200, 14500, 17800, 21100, 24400,
    27700, 31000, 34300, 37600, 40900
])

model = LinearRegression()
model.fit(X,y)
print("Model trained successfully.")

APM = int(input("Enter Active Premium Members:"))
expence = int(input("Enter Monthly Digital Marketing Expenditure ($):"))
gym = np.array([[APM,expence]])

revenue = model.predict(gym)[0]
print(f"Gym Active Premium Members: {gym[0,0]:.2f}")
print(f"Gym Monthly Digital Marketing Expenditure ($): {gym[0,1]:.2f}$")
print(f"Gym Revenue: {revenue:.2f}$")