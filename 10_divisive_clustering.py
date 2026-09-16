import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "Country": [
        "India", "China", "USA", "Germany",
        "Japan", "Brazil", "Canada", "Nigeria",
        "Switzerland", "Bangladesh", "Australia", "South Africa"
    ],

    # Column 1 = GDP per Capita (USD)
    # Column 2 = Life Expectancy (Years)
    # Column 3 = Internet Usage (%)

    "GDP_per_capita": [
        2700, 13000, 85000, 55000,
        34000, 11000, 53000, 1100,
        100000, 2700, 65000, 6500
    ],

    "Life_expectancy": [
        67, 78, 77, 81,
        84, 76, 82, 54,
        84, 73, 83, 62
    ],

    "Internet_usage": [
        55, 76, 97, 92,
        87, 81, 94, 36,
        96, 45, 97, 75
    ]
}

df = pd.DataFrame(data)

X = df[
    [
        "GDP_per_capita",
        "Life_expectancy",
        "Internet_usage"
    ]
]

X_scaled = StandardScaler().fit_transform(X)


def divisive_clustering(X, number_of_clusters=4):

    clusters = {
        0: list(range(len(X)))
    }

    next_cluster_id = 1

    while len(clusters) < number_of_clusters:

        largest_cluster_id = max(
            clusters,
            key=lambda cluster_id: len(clusters[cluster_id])
        )

        indexes = clusters[largest_cluster_id]
        cluster_data = X[indexes]

        model = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(cluster_data)

        cluster_1 = []
        cluster_2 = []

        for label, index in zip(labels, indexes):

            if label == 0:
                cluster_1.append(index)
            else:
                cluster_2.append(index)

        del clusters[largest_cluster_id]

        clusters[next_cluster_id] = cluster_1
        next_cluster_id += 1

        clusters[next_cluster_id] = cluster_2
        next_cluster_id += 1

    return clusters


clusters = divisive_clustering(
    X_scaled,
    number_of_clusters=4
)


cluster_labels = {}

for cluster_id, indexes in clusters.items():

    for index in indexes:
        cluster_labels[index] = cluster_id

df["Cluster"] = pd.Series(cluster_labels)



summary = df.groupby("Cluster")[
    [
        "GDP_per_capita",
        "Life_expectancy",
        "Internet_usage"
    ]
].mean()


cluster_names = {}

for cluster_id, row in summary.iterrows():

    gdp = row["GDP_per_capita"]
    life = row["Life_expectancy"]
    internet = row["Internet_usage"]

    if gdp >= 50000 and life >= 80 and internet >= 90:
        name = "Highly Developed"

    elif gdp >= 10000 and life >= 75 and internet >= 70:
        name = "Developing / Upper Middle"

    elif life < 65 or internet < 60:
        name = "Low Development"

    else:
        name = "Moderate Development"

    cluster_names[cluster_id] = name

df["ClusterName"] = df["Cluster"].map(cluster_names)


print("\n========== FINAL DATASET ==========")

print(
    df[
        [
            "Country",
            "GDP_per_capita",
            "Life_expectancy",
            "Internet_usage",
            "ClusterName"
        ]
    ].to_string(index=False)
)

plt.figure(figsize=(11, 7))

for cluster_id, cluster_name in cluster_names.items():

    cluster_data = df[df["Cluster"] == cluster_id]

    plt.scatter(
        cluster_data["GDP_per_capita"],
        cluster_data["Life_expectancy"],
        label=cluster_name
    )

    # Country names on chart
    for _, row in cluster_data.iterrows():

        plt.annotate(
            row["Country"],
            (
                row["GDP_per_capita"],
                row["Life_expectancy"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )


plt.xlabel("GDP per Capita ($)")
plt.ylabel("Life Expectancy (Years)")
plt.title("Divisive Clustering of Countries")
plt.legend(title="Cluster")

plt.show()