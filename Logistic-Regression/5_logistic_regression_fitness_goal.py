import numpy as np
from sklearn.linear_model import LogisticRegression


# | Variable | Meaning                        |
# | -------- | ------------------------------ |
# |  0       | Active Minutes                 |
# |  1       | Calories Consumed in Thousands |

X = np.array([
    [10, 2.5], [15, 2.4], [20, 2.6], [25, 2.5], [30, 2.7],
    [35, 2.6], [40, 2.8], [45, 2.7], [50, 2.9], [55, 2.8],

    [12, 2.3], [18, 2.5], [22, 2.4], [28, 2.6], [32, 2.5],
    [38, 2.7], [42, 2.6], [48, 2.8], [52, 2.7], [58, 2.9],

    [20, 2.8], [25, 2.9], [30, 3.0], [35, 2.8], [40, 3.1],
    [45, 3.0], [50, 3.2], [55, 3.1], [60, 3.3], [65, 3.2],

    [25, 2.2], [30, 2.4], [35, 2.5], [40, 2.7], [45, 2.8],
    [50, 2.9], [55, 3.0], [60, 3.1], [65, 3.2], [70, 3.3],

    [30, 2.5], [35, 2.6], [40, 2.8], [45, 2.9], [50, 3.0],
    [55, 3.1], [60, 3.2], [65, 3.3], [70, 3.4], [75, 3.5],

    [35, 2.4], [40, 2.6], [45, 2.7], [50, 2.9], [55, 3.0],
    [60, 3.1], [65, 3.2], [70, 3.3], [75, 3.4], [80, 3.5],

    [40, 2.6], [45, 2.8], [50, 2.9], [55, 3.0], [60, 3.1],
    [65, 3.2], [70, 3.3], [75, 3.4], [80, 3.5], [85, 3.6],

    [45, 2.5], [50, 2.7], [55, 2.8], [60, 3.0], [65, 3.1],
    [70, 3.2], [75, 3.3], [80, 3.4], [85, 3.5], [90, 3.6],

    [50, 2.8], [55, 2.9], [60, 3.0], [65, 3.2], [70, 3.3],
    [75, 3.4], [80, 3.5], [85, 3.6], [90, 3.7], [95, 3.8],

    [55, 2.7], [60, 2.9], [65, 3.0], [70, 3.2], [75, 3.3],
    [80, 3.4], [85, 3.5], [90, 3.6], [95, 3.7], [100, 3.8]
])

# 0 = Miss Daily Calorie Goal
# 1 = Achieve Daily Calorie Goal

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

def predict_fitness(person):

    prediction = model.predict(person)
    probability = model.predict_proba(person)
    
    result = "Achieve Daily Calorie Goal " if prediction[0]==1 else " Miss Daily Calorie Goal"
    
    print(f"Active Minutes : {person[0,0]}")
    print(f"Calories Consumed in Thousands:  {person[0,1]*1000}")
    print(f"Prediction: {result}")
        
    print(f"Probability of  Miss Daily Calorie Goal:{probability[0, 0]:.2%}")
        
    print(f"Probability of Achieve Daily Calorie Goal :{probability[0, 1]:.2%}\n")
    
user_1 = np.array([[20, 2.5]])
user_2 = np.array([[40, 2.8]])
user_3 = np.array([[60, 3.1]])
user_4 = np.array([[75, 3.4]])
print("...user-1...")
predict_fitness(user_1)
print("...user-2...")
predict_fitness(user_2)
print("...user-3...")
predict_fitness(user_3)
print("...user-4...")
predict_fitness(user_4)

accuracy = model.score(X, y)
print(f"\nTraining Accuracy: {accuracy:.2%}")