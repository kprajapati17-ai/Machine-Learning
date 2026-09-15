import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage,dendrogram


data = {
    "Customer": [
        "C1", "C2", "C3",
        "C4", "C5", "C6",
        "C7", "C8", "C9",
        "C10", "C11", "C12"
    ],

    "Income": [
        1500000, 1800000, 2000000,
        4500000, 4800000, 5000000,
        8000000, 8500000, 9000000,
        3000000, 3300000, 13500000
    ],

    "Purchases": [
        300000, 400000, 500000,
        1000000, 1100000, 1200000,
        2000000, 2100000, 2200000,
        700000, 800000, 8000000
    ]
}

df = pd.DataFrame(data)


# Column 1 = Income
# Column 2 = Purchases
X = df[["Income","Purchases"]]

X_scaled = StandardScaler().fit_transform(X)

linkage_matrix = linkage(X_scaled,method='ward')

plt.figure(figsize=(10,6))
dendrogram(
    linkage_matrix,
    labels=df["Customer"].values
)

plt.xlabel("Customer")
plt.ylabel("Distance")
plt.title("Customer Hierarchical Clustering")

plt.show()

model = AgglomerativeClustering(n_clusters=4,linkage='ward')

names = {
    0: "Low Value Customer",
    1: "Medium Value Customer",
    2: "High Value Customer",
    3: "Premium Customer"
}

df["Cluster"]=model.fit_predict(X_scaled)
df['ClusterName']= df["Cluster"].map(names)
print("\nFinal Customer Clusters:")
print(df)