import numpy as np
from sklearn.linear_model import LogisticRegression

# Features:
# Column 1 = BMI
# Column 2 = Age
X = np.array([
    [30, 30],
    [31, 35],
    [32, 40],
    [33, 41],
    [34, 42],
    [35, 45],
    [33, 50],
    [28, 28],
    [27, 30],
    [26, 30],
    [25, 51],
    [24, 52],
    [23, 55]
])

# Target:
# 0 = No Diabetes
# 1 = Diabetes
y = np.array([
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
])


model  = LogisticRegression()

model.fit(X,y)

print("Model trained successfully.")


def predict_diabetes(person):

    # person = np.array([[bmi,age]])

    prediction = model.predict(person)
    probability = model.predict_proba(person)

    result = 'diabetes' if prediction[0]==1 else 'no diabetes'

    print(f"\nBMI: {person[0,0]}")
    print(f"Age: {person[0,1]}")
    print(f"Prediction: {result}")
    print(f"Probability of No Diabetes: {probability[0,0]:.2%}")
    print(f"Probability of Diabetes: {probability[0,1]:.2%}")

hanshraj_hathi =np.array([[40,45]])
print("hanshraj hathi ")
predict_diabetes(hanshraj_hathi)

tarak_maheta =np.array([[28,50]])
print("tarak maheta ")
predict_diabetes(tarak_maheta)

accuracy = model.score(X, y)

print(f"\nTraining Accuracy: {accuracy:.2%}")
