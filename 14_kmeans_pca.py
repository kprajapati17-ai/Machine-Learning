import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

X = StandardScaler().fit_transform(df[features])

pca = PCA(n_components=2)
X_pca=pca.fit_transform(X)

print(f"PC-1 Variance:{pca.explained_variance_ratio_[0]:.2%}")
print(f"PC-2 Variance:{pca.explained_variance_ratio_[1]:.2%}")
print(f"Total Variance:{pca.explained_variance_ratio_.sum():.2%}")

inertia =[]

for k in range(1,7):
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_pca)
    inertia.append(model.inertia_)

model = KMeans(n_clusters=3,random_state=42,n_init=10)
df["cluster"] = model.fit_predict(X_pca)

summary = df.groupby("cluster")[features].mean()
name={}

for cluster,row in summary.iterrows():

    if(row["Activity"]>summary["Activity"].median() and  row["BMI"] < summary["BMI"].median()):
        name[cluster] = "Active Pattern"
    elif(row["BMI"]>summary["BMI"].median() and  row["Glucose"] < summary["Glucose"].median()):
        name[cluster] = "Higher Risk Pattern"
    else:
        name[cluster] = "Moderate Pattern"

df['clustername'] = df["cluster"].map(name)

print(
    df[
        ["Patient", "cluster", "clustername"]
    ]
)