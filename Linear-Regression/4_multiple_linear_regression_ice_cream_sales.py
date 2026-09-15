import numpy as np
from sklearn.linear_model import LinearRegression

# Column 1 = Maximum Daily Temperature (°C)
# Column 2 = Estimated Daily Foot Traffic (people)

X = np.array([
    [20, 500],    # 1
    [21, 600],    # 2
    [22, 700],    # 3
    [23, 800],    # 4
    [24, 900],    # 5
    [25, 1000],   # 6
    [26, 1100],   # 7
    [27, 1200],   # 8
    [28, 1300],   # 9
    [29, 1400],   # 10

    [20, 700],    # 11
    [22, 900],    # 12
    [24, 1100],   # 13
    [26, 1300],   # 14
    [28, 1500],   # 15
    [30, 1700],   # 16
    [32, 1900],   # 17
    [34, 2100],   # 18
    [36, 2300],   # 19
    [38, 2500],   # 20

    [21, 800],    # 21
    [23, 1000],   # 22
    [25, 1200],   # 23
    [27, 1400],   # 24
    [29, 1600],   # 25
    [31, 1800],   # 26
    [33, 2000],   # 27
    [35, 2200],   # 28
    [37, 2400],   # 29
    [39, 2600]    # 30
])

# y = Number of Ice Cream Cones Sold per Day

y = np.array([
    120,   # 1
    145,   # 2
    170,   # 3
    195,   # 4
    220,   # 5
    250,   # 6
    280,   # 7
    310,   # 8
    345,   # 9
    380,   # 10

    150,   # 11
    190,   # 12
    235,   # 13
    280,   # 14
    330,   # 15
    390,   # 16
    450,   # 17
    510,   # 18
    575,   # 19
    640,   # 20

    175,   # 21
    220,   # 22
    270,   # 23
    325,   # 24
    385,   # 25
    450,   # 26
    520,   # 27
    595,   # 28
    670,   # 29
    750    # 30
])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X rows:", len(X))
print("y values:", len(y))

model = LinearRegression()
model.fit(X, y)

temperature = float(
    input("\nEnter maximum temperature (°C): ")
)

foot_traffic = int(
    input("Enter estimated foot traffic: ")
)

new_day = np.array([
    [temperature, foot_traffic]
])


predicted_sales = model.predict(new_day)[0]

print("\n========== PREDICTION ==========")
print(f"Maximum Temperature: {temperature:.1f}°C")
print(f"Estimated Foot Traffic: {foot_traffic}")
print(f"Predicted Ice Cream Sales: {predicted_sales:.0f} cones")

