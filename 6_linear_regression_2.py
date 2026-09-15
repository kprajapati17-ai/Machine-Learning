import numpy as np
from sklearn.linear_model import LinearRegression

# Features:
# Column 1 = House Size (hundreds of sq. ft.)
# Column 2 = Number of Bedrooms

X = np.array([
    [120, 2],
    [150, 3],
    [180, 3],
    [200, 4],
    [250, 5]
])


# Target:
# House Price in $1,000s

y = np.array([
    150,
    200,
    240,
    290,
    350
])

model = LinearRegression()
model.fit(X,y)
print("Model trained successfully.")

house1 = np.array([[400,7]])
prediction1 = model.predict(house1)[0]
print(f"House 1 predicted price: ${prediction1:.2f}K")

house2 = np.array([[200,4]])
prediction2 = model.predict(house2)[0]
print(f"\nHouse 2 predicted price: ${prediction2:.2f}K")


print("\nModel Information:")
print(f"House Size Coefficient: {model.coef_[0]:.2f}")
print(f"Bedroom Coefficient: {model.coef_[1]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")