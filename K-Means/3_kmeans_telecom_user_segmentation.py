import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# 1. Dataset
# ==========================================

# Column 1 = Monthly Data Usage (GB)
# Column 2 = Domestic Call Minutes
# Column 3 = International SMS Count
# Column 4 = OffPeak Data Usage Ratio

data = {
    "User": [
        "U01", "U02", "U03", "U04", "U05",
        "U06", "U07", "U08", "U09", "U10",
        "U11", "U12", "U13", "U14", "U15",
        "U16", "U17", "U18", "U19", "U20",
        "U21", "U22", "U23", "U24", "U25",
        "U26", "U27", "U28", "U29", "U30"
    ],

    "DataUsageGB": [
        45, 50, 55, 48, 60,
        52, 58, 62, 47, 65,

        8, 10, 12, 9, 14,
        11, 13, 15, 10, 16,

        25, 28, 30, 27, 32,
        29, 35, 31, 26, 34
    ],

    "CallMinutes": [
        80, 90, 70, 100, 85,
        95, 75, 110, 88, 82,

        900, 950, 1000, 850, 1100,
        980, 1050, 1150, 920, 1080, 1200,

        300, 320, 280, 350, 310,
        330, 290, 360, 340, 305
    ],

    "InternationalSMS": [
        10, 12, 8, 15, 9,
        11, 7, 14, 10, 13,

        65, 70, 80, 60, 85,
        75, 90, 68, 82, 95,

        20, 22, 18, 25, 21,
        24, 19, 27, 23, 20
    ],

    "OffPeakRatio": [
        0.35, 0.38, 0.32, 0.40, 0.36,
        0.34, 0.39, 0.37, 0.33, 0.41,

        0.30, 0.28, 0.32, 0.27, 0.31,
        0.29, 0.33, 0.30, 0.28, 0.34,

        0.75, 0.82, 0.78, 0.80, 0.85,
        0.77, 0.88, 0.81, 0.76, 0.84
    ]
}


# ==========================================
# 2. Create DataFrame
# ==========================================

df = pd.DataFrame(data)


# ==========================================
# 3. Select Features
# ==========================================

X = df[
    [
        "DataUsageGB",
        "CallMinutes",
        "InternationalSMS",
        "OffPeakRatio"
    ]
]


# ==========================================
# 4. Standardize Features
# ==========================================

X_scaled = StandardScaler().fit_transform(X)


# ==========================================
# 5. Find K using Elbow Method
# ==========================================

inertia = []

for k in range(1, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)


# ==========================================
# 6. Elbow Graph
# ==========================================

plt.plot(
    range(1, 7),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method - Telecom User Segmentation")

plt.show()


# ==========================================
# 7. Final K-Means Model
# ==========================================

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)


# ==========================================
# 8. Find Cluster Average
# ==========================================

summary = df.groupby("Cluster")[
    [
        "DataUsageGB",
        "CallMinutes",
        "InternationalSMS",
        "OffPeakRatio"
    ]
].mean()


# ==========================================
# 9. Assign Cluster Names
# ==========================================

cluster_names = {}

for cluster, row in summary.iterrows():

    if (
        row["DataUsageGB"] > 40
        and row["CallMinutes"] < 200
    ):
        cluster_names[cluster] = "Heavy Data Users"

    elif (
        row["CallMinutes"] > 700
        and row["InternationalSMS"] > 50
    ):
        cluster_names[cluster] = "Calling Users"

    else:
        cluster_names[cluster] = "Night Users"


# ==========================================
# 10. Add Cluster Name
# ==========================================

df["ClusterName"] = df["Cluster"].map(cluster_names)


# ==========================================
# 11. Display Final Result
# ==========================================

print("\n===== TELECOM USER SEGMENTATION =====")

print(
    df[
        [
            "User",
            "DataUsageGB",
            "CallMinutes",
            "InternationalSMS",
            "OffPeakRatio",
            "ClusterName"
        ]
    ]
)


# ==========================================
# 12. Display Cluster Summary
# ==========================================

print("\n===== CLUSTER SUMMARY =====")
print(summary)