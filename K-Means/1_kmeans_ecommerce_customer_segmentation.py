import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dataset
# Area/Customer ID = Customer identifier
# AnnualSpending = Annual spending in dollars
# PurchaseFrequency = Number of orders per year
# ReturnRate = Average product return rate (%)

data = {
    "Customer": [
        "C01", "C02", "C03", "C04", "C05",
        "C06", "C07", "C08", "C09", "C10",
        "C11", "C12", "C13", "C14", "C15",
        "C16", "C17", "C18", "C19", "C20",
        "C21", "C22", "C23", "C24", "C25",
        "C26", "C27", "C28", "C29", "C30",
        "C31", "C32", "C33", "C34", "C35",
        "C36", "C37", "C38", "C39", "C40"
    ],

    "AnnualSpending": [
        240000, 280000, 320000, 260000, 350000,
        300000, 380000, 270000, 330000, 290000,

        250000, 310000, 360000, 280000, 340000,
        300000, 390000, 270000, 325000, 355000,

        85000, 105000, 120000, 95000, 135000,
        110000, 145000, 100000, 125000, 115000,

        35000, 45000, 60000, 30000, 70000,
        50000, 80000, 40000, 65000, 55000
    ],

    "PurchaseFrequency": [
        72, 80, 76, 68, 85,
        74, 90, 70, 82, 78,

        65, 72, 78, 60, 70,
        68, 80, 62, 74, 76,

        52, 58, 64, 55, 68,
        60, 70, 56, 62, 59,

        8, 12, 15, 6, 18,
        10, 20, 9, 14, 11
    ],

    "ReturnRate": [
        4, 3, 5, 4, 3,
        6, 4, 5, 4, 3,

        24, 28, 22, 26, 30,
        25, 27, 23, 29, 24,

        5, 6, 4, 7, 5,
        6, 4, 5, 6, 4,

        7, 8, 6, 5, 9,
        7, 8, 6, 7, 5
    ]
}

df = pd.DataFrame(data)

print(df)
print("\nShape:", df.shape)

X = df[["AnnualSpending","PurchaseFrequency","ReturnRate"]]

scaler = StandardScaler()
x_scaler = scaler.fit_transform(X)

model=KMeans(n_clusters=4,random_state=45,n_init=10)
df["cluster"] =model.fit_predict(x_scaler)


names = {
    0: "VIP Customers",
    1: "HighRisk Customers",
    2: "Regular Budget Customers",
    3: "Occasional Customers"
}

df["ClusterName"] = df["cluster"].map(names)

print("\nFinal Clustered Data:")
print(df)