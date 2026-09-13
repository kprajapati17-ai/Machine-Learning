import numpy as np
from sklearn.linear_model import LogisticRegression


study_hours = np.array([
    [2],
    [4],
    [6],
    [8],
    [9],
    [10],
    [11],
    [12]
])

# Target: 0 = Fail, 1 = Pass
result = np.array([0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(study_hours,result)


print("Model trained successfully.")


# Predict result for 7 hours of study
hours = np.array([[7]])

prediction = model.predict(hours)
probability = model.predict_proba(hours)

print(f"Study Hours :{hours[0][0]}")
print(f"prediction :{'pass' if prediction[0]==1 else 'fail' }")
print(f"Probability of Fail: {probability[0, 0]:.2f}")
print(f"Probability of Pass: {probability[0, 1]:.2f}")


# Predict result for 3.5 hours of study
hours = np.array([[3.5]])

prediction = model.predict(hours)
probability = model.predict_proba(hours)

print(f"\nStudy hours: {hours[0, 0]}")
print(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"Probability of Fail: {probability[0, 0]:.2f}")
print(f"Probability of Pass: {probability[0, 1]:.2f}")

accuracy = model.score(study_hours,result)

print(f"\nTraining Accuracy: {accuracy:.2%}")    