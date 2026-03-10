import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import KBinsDiscretizer
import os

# Set base directory
base_dir = "."
data_path = os.path.join(base_dir, "dummy-data", "vehicles_ml_dataset.csv")

# Load data
df = pd.read_csv(data_path)
income = df["estimated_income"].to_numpy().reshape(-1, 1)
price = df["selling_price"].to_numpy().reshape(-1, 1)

# Parameters from optimized train_cluster.py
n_bins = 15
income_weight = 20
n_clusters = 6

# Process
kbd_i = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
kbd_p = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")

income_b = kbd_i.fit_transform(income)
price_b = kbd_p.fit_transform(price)

X = np.column_stack([income_b * income_weight, price_b])
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
labels = kmeans.fit_predict(X)

# Calculate silhouette
score = silhouette_score(X, labels)
print(f"SILHOUETTE_SCORE: {score:.4f}")

# Calculate CV for verification
df['cluster'] = labels
for cluster in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster]
    if len(cluster_data) > 1:
        inc_cv = (cluster_data['estimated_income'].std() / cluster_data['estimated_income'].mean()) * 100
        pri_cv = (cluster_data['selling_price'].std() / cluster_data['selling_price'].mean()) * 100
        print(f"Cluster {cluster}: Income CV={inc_cv:.2f}%, Price CV={pri_cv:.2f}%")
