import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("patient_health_data.csv")

features = [
    "Age",
    "Systolic_BP",
    "Diastolic_BP",
    "Cholesterol",
    "Glucose",
    "BMI",
    "Heart_Rate",
    "Activity"
]

X = df[features]

x_scaler = StandardScaler().fit_transform(X)

inertia =[]

for k in range(1,7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(x_scaler)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,7),inertia,marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()


model = KMeans(n_clusters=3,random_state=42,n_init=10)
df["cluster"] = model.fit_predict(x_scaler)

print(df)

