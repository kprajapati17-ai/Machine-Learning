import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Dataset
# ==========================================

# Column 1 = Price per Square Foot ($)
# Column 2 = Distance from Transit Hub (km)
# Column 3 = School Rating (1-10)

data = {
    "Area": [
        "A01", "A02", "A03", "A04", "A05",
        "A06", "A07", "A08", "A09", "A10",
        "A11", "A12", "A13", "A14", "A15",
        "A16", "A17", "A18", "A19", "A20",
        "A21", "A22", "A23", "A24", "A25",
        "A26", "A27", "A28", "A29", "A30",
        "A31", "A32", "A33", "A34", "A35",
        "A36", "A37", "A38", "A39", "A40"
    ],

    "PricePerSqFt": [
        950, 1100, 1200, 1000, 1150,
        1050, 1250, 980, 1180, 1080,

        900, 1000, 1100, 950, 1050,
        1150, 1080, 990, 1120, 1020,

        650, 700, 750, 680, 720,
        760, 690, 730, 710, 740,

        350, 400, 450, 380, 420,
        460, 370, 430, 390, 410
    ],

    "TransitDistance": [
        0.5, 0.8, 1.0, 0.6, 0.9,
        0.7, 1.1, 0.4, 0.8, 0.6,

        3.0, 3.5, 4.0, 3.2, 4.5,
        3.8, 4.2, 3.6, 4.8, 3.4,

        2.0, 2.5, 3.0, 2.2, 2.8,
        3.1, 2.4, 2.7, 2.1, 2.9,

        6.0, 6.5, 7.0, 6.2, 7.5,
        6.8, 7.2, 6.6, 7.8, 7.0
    ],

    "SchoolRating": [
        8.5, 8.8, 9.0, 8.6, 9.1,
        8.7, 9.3, 8.4, 9.0, 8.9,

        9.5, 9.8, 9.2, 9.6, 9.9,
        9.4, 9.7, 9.3, 9.8, 9.6,

        6.0, 6.5, 6.8, 5.8, 6.3,
        6.7, 6.1, 6.6, 6.4, 5.9,

        3.5, 4.0, 4.2, 3.8, 4.5,
        3.2, 4.1, 3.7, 4.3, 3.9
    ]
}


# ==========================================
# 2. Create DataFrame
# ==========================================

df = pd.DataFrame(data)


# ==========================================
# 3. Select Features
# ==========================================

# Column 1 = Price per Square Foot
# Column 2 = Distance from Transit Hub
# Column 3 = School Rating

X = df[
    [
        "PricePerSqFt",
        "TransitDistance",
        "SchoolRating"
    ]
]


# ==========================================
# 4. Standardize Features
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# 5. K-Means Model
# ==========================================

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)


# ==========================================
# 6. Find Cluster Characteristics
# ==========================================

summary = df.groupby("Cluster")[
    [
        "PricePerSqFt",
        "TransitDistance",
        "SchoolRating"
    ]
].mean()


# ==========================================
# 7. Give Names to Clusters
# ==========================================

cluster_names = {}

for cluster, row in summary.iterrows():

    price = row["PricePerSqFt"]
    distance = row["TransitDistance"]
    school = row["SchoolRating"]

    if price >= 900 and distance <= 1.5:
        cluster_names[cluster] = "Premium Areas"

    elif price >= 900 and school >= 9:
        cluster_names[cluster] = "Family Areas"

    elif price >= 600 and price < 900:
        cluster_names[cluster] = "Average Areas"

    else:
        cluster_names[cluster] = "Budget Areas"


# ==========================================
# 8. Add Cluster Names
# ==========================================

df["ClusterName"] = df["Cluster"].map(cluster_names)


# ==========================================
# 9. Display Final Result
# ==========================================

print("\n========== FINAL HOUSING SEGMENTATION ==========")

print(
    df[
        [
            "Area",
            "PricePerSqFt",
            "TransitDistance",
            "SchoolRating",
            "ClusterName"
        ]
    ].to_string(index=False)
)


# ==========================================
# 10. Display Cluster Summary
# ==========================================

print("\n========== CLUSTER SUMMARY ==========")

summary["ClusterName"] = summary.index.map(cluster_names)

print(summary)