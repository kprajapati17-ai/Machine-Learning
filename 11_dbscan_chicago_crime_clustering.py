import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

url = (
    "https://data.cityofchicago.org/resource/"
    "ijzp-q8t2.csv?$limit=5000"
)

df = pd.read_csv(url)
data = df[
    ["latitude", "longitude"]
].copy()


data["latitude"]= pd.to_numeric(data["latitude"],errors="coerce")
data["longitude"]= pd.to_numeric(data["longitude"],errors="coerce")


data = data.dropna()
model = DBSCAN(eps=0.005,min_samples=10)
data["Cluster"]= model.fit_predict(data[["latitude", "longitude"]])

print("\n===== FINAL DATA =====")
print(
    data[
        [
            "latitude",
            "longitude",
            "Cluster",
        ]
    ].head(100)
)

plt.figure(figsize=(10, 8))

plt.scatter(
    data["longitude"],
    data["latitude"],
    s=10,
    c=data["Cluster"]
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Chicago Crime Clustering using DBSCAN")

plt.show()