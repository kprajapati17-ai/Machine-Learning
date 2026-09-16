import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_scaler)

pca_df = pd.DataFrame(
    pca_result,
    columns=["PC-1", "PC-2"]
)
pca_df["Patient"] = df["Patient"]

print(f"PC-1 Variance:{pca.explained_variance_ratio_[0]:.2%}")
print(f"PC-2 Variance:{pca.explained_variance_ratio_[1]:.2%}")
print(f"Total Variance Explained:{pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(10,8))

plt.scatter(
    pca_df["PC-1"],
    pca_df["PC-2"],
    s=100
)
for _, row in pca_df.iterrows():

    plt.annotate(
        row["Patient"],
        (row["PC-1"], row["PC-2"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.title("PCA - Patient Health Analysis")

plt.axhline(0)
plt.axvline(0)

plt.show()