import numpy as np
from sklearn.linear_model import LogisticRegression

# X:
# Column 1 = Operating Temperature in °C
# Column 2 = Vibration Level mm/s (millimeters per second)


X = np.array([
    [55, 1.2],
    [58, 1.4],
    [60, 1.5],
    [62, 1.6],
    [64, 1.7],
    [65, 1.8],
    [67, 1.9],
    [68, 2.0],
    [70, 2.1],
    [72, 2.2],

    [56, 1.3],
    [59, 1.5],
    [61, 1.6],
    [63, 1.7],
    [65, 1.9],
    [66, 2.0],
    [68, 2.1],
    [70, 2.2],
    [71, 2.3],
    [73, 2.4],

    [60, 1.4],
    [62, 1.6],
    [64, 1.8],
    [66, 1.9],
    [68, 2.0],
    [70, 2.2],
    [72, 2.3],
    [74, 2.5],
    [76, 2.6],
    [78, 2.7],

    [65, 1.8],
    [67, 2.0],
    [69, 2.1],
    [71, 2.3],
    [73, 2.4],
    [75, 2.5],
    [77, 2.7],
    [79, 2.8],
    [81, 2.9],
    [83, 3.0],

    [68, 2.0],
    [70, 2.2],
    [72, 2.4],
    [74, 2.5],
    [76, 2.7],
    [78, 2.8],
    [80, 3.0],
    [82, 3.1],
    [84, 3.2],
    [86, 3.3],

    [70, 2.2],
    [72, 2.4],
    [74, 2.6],
    [76, 2.8],
    [78, 2.9],
    [80, 3.1],
    [82, 3.2],
    [84, 3.4],
    [86, 3.5],
    [88, 3.6],

    [72, 2.5],
    [74, 2.7],
    [76, 2.9],
    [78, 3.0],
    [80, 3.2],
    [82, 3.3],
    [84, 3.5],
    [86, 3.6],
    [88, 3.8],
    [90, 3.9],

    [75, 2.8],
    [77, 3.0],
    [79, 3.2],
    [81, 3.4],
    [83, 3.5],
    [85, 3.7],
    [87, 3.8],
    [89, 4.0],
    [91, 4.1],
    [93, 4.2],

    [78, 3.0],
    [80, 3.2],
    [82, 3.4],
    [84, 3.6],
    [86, 3.7],
    [88, 3.9],
    [90, 4.0],
    [92, 4.2],
    [94, 4.3],
    [96, 4.5],

    [80, 3.2],
    [82, 3.4],
    [84, 3.6],
    [86, 3.8],
    [88, 4.0],
    [90, 4.1],
    [92, 4.3],
    [94, 4.5],
    [96, 4.6],
    [98, 4.8]
])

# 0 = Operate Normally
# 1 = Machine Fail

y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 1,

    0, 0, 0, 0, 0,
    0, 0, 1, 1, 1,

    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,

    0, 0, 0, 1, 1,
    1, 1, 1, 1, 1,

    0, 0, 1, 1, 1,
    1, 1, 1, 1, 1,

    0, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])

print("X shape:", X.shape)
print("y shape:", y.shape)


model = LogisticRegression()
model.fit(X,y)

print("\nModel trained successfully.")


def predict_failure(machine):

    prediction = model.predict(machine)
    probability = model.predict_proba(machine)

    result = "Machine Operate Normally " if prediction[0]==0 else "Machine Fail.."

    print(f"\nOperating Temperature in °C : {machine[0,0]}")
    print(f"Vibration Level mm/s (millimeters per second):  {machine[0,1]}")
    print(f"Prediction: {result}")
    
    print(f"Probability of Machine Operate Normally:{probability[0, 0]:.2%}")
    
    print(f"Probability of Machine Fail..:{probability[0, 1]:.2%}")

machine_2 = np.array([[72, 2.4]])
print("...machine2...")
predict_failure(machine_2)

machine_3 = np.array([[82, 3.3]])
print("...machine3...")
predict_failure(machine_3)

accuracy = model.score(X, y)
print(f"\nTraining Accuracy: {accuracy:.2%}")