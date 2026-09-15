import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "Area": [
        "A", "B", "C", "D",
        "E", "F", "G", "H",
        "I", "J", "K", "L"
    ],

    "Direction": [
        "North", "North", "North", "North",
        "East", "East", "East", "East",
        "South", "South", "South", "South"
    ],

    "Distance": [
        2, 3, 4, 5,
        10, 11, 12, 13,
        20, 21, 22, 23
    ],

    "DailyOrders": [
        95, 90, 100, 85,
        55, 60, 50, 58,
        20, 25, 18, 22
    ]
}


df = pd.DataFrame(data)
print("Original Data:")
print(df)


X = df[[
    "Distance",
    "DailyOrders"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia =[]

for k in range(1,7):
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 7),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()

model = KMeans(n_clusters=3,random_state=45,n_init=10)
model.fit(X_scaled)

labels= model.labels_
df["cluster"] =labels

print("\nFinal Clustered Data:")
print(df)