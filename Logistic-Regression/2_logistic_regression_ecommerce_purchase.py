import numpy as np
from sklearn.linear_model import LogisticRegression

# Column 1 = Time Spent on Site (minutes)
# Column 2 = Number of Pages Viewed

X = np.array([
    [2, 2],
    [3, 3],
    [4, 3],
    [5, 4],
    [6, 4],
    [7, 5],
    [8, 5],
    [9, 6],
    [10, 6],
    [11, 7],

    [3, 2],
    [4, 4],
    [5, 3],
    [6, 5],
    [7, 4],
    [8, 6],
    [9, 5],
    [10, 7],
    [11, 6],
    [12, 8],

    [5, 5],
    [6, 6],
    [7, 5],
    [8, 7],
    [9, 6],
    [10, 8],
    [11, 7],
    [12, 9],
    [13, 8],
    [14, 10],

    [6, 4],
    [7, 6],
    [8, 6],
    [9, 7],
    [10, 7],
    [11, 8],
    [12, 8],
    [13, 9],
    [14, 9],
    [15, 10],

    [8, 5],
    [9, 7],
    [10, 8],
    [11, 9],
    [12, 9],
    [13, 10],
    [14, 10],
    [15, 11],
    [16, 11],
    [17, 12],

    [10, 6],
    [11, 8],
    [12, 10],
    [13, 9],
    [14, 11],
    [15, 10],
    [16, 12],
    [17, 11],
    [18, 13],
    [19, 12],

    [12, 7],
    [13, 9],
    [14, 10],
    [15, 11],
    [16, 12],
    [17, 13],
    [18, 12],
    [19, 14],
    [20, 13],
    [21, 15],

    [14, 8],
    [15, 10],
    [16, 11],
    [17, 12],
    [18, 13],
    [19, 14],
    [20, 15],
    [21, 14],
    [22, 16],
    [23, 15],

    [16, 9],
    [17, 11],
    [18, 12],
    [19, 13],
    [20, 14],
    [21, 15],
    [22, 16],
    [23, 17],
    [24, 16],
    [25, 18],

    [18, 10],
    [19, 12],
    [20, 13],
    [21, 14],
    [22, 15],
    [23, 16],
    [24, 17],
    [25, 18],
    [26, 19],
    [28, 20]
])


# 0 = Abandon Cart
# 1 = Complete Purchase

y = np.array([
    # 1 - 10
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    # 11 - 20
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    # 21 - 30
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 1,

    # 31 - 40
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 1,

    # 41 - 50
    0, 0, 0, 0, 0,
    0, 1, 1, 1, 1,

    # 51 - 60
    0, 0, 0, 1, 1,
    1, 1, 1, 1, 1,

    # 61 - 70
    0, 0, 1, 1, 1,
    1, 1, 1, 1, 1,

    # 71 - 80
    0, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    # 81 - 90
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    # 91 - 100
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])


print("X shape:", X.shape)
print("y shape:", y.shape)

print("Number of shoppers:", len(X))
print("Number of target values:", len(y))


model = LogisticRegression()

model.fit(X,y)

def predict_purchase(shopper):

    prediction = model.predict(shopper)
    probability = model.predict_proba(shopper)

    result = ( "Complete Purchase" if prediction[0] == 1 else "Abandon Cart")

    print(f"\nTime Spent on Site (minutes): {shopper[0, 0]} ")
    print(f"Pages Viewed: {shopper[0, 1]}")
    print(f"Prediction: {result}")
    
    print(f"Probability of Abandon Cart: {probability[0, 0]:.2%}")
        
    print(f"Probability of  Complete Purchase: {probability[0, 1]:.2%}")


shopper1 = np.array([[5,4]])
print("......shopper-1.........")
predict_purchase(shopper1)

shopper2 = np.array([[22,10]])
print("......shopper-2.........")
predict_purchase(shopper2)

accuracy = model.score(X, y)
print(f"\nTraining Accuracy: {accuracy:.2%}")