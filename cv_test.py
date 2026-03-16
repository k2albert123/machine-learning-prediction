import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def analyze_cv(n_bins=6, income_weight=1000, n_clusters=6, strategy="kmeans"):
    csv_path = "dummy-data/vehicles_ml_dataset.csv"
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    income = df["estimated_income"].to_numpy().reshape(-1, 1)
    price = df["selling_price"].to_numpy().reshape(-1, 1)
    
    kbd_i = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    kbd_p = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    
    income_b = kbd_i.fit_transform(income)
    price_b = kbd_p.fit_transform(price)
    
    X = np.column_stack([income_b * income_weight, price_b])
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=42)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels)
    df["cluster"] = labels
    
    cvs = []
    for c in np.unique(labels):
        cluster_data = df[df["cluster"] == c]
        if len(cluster_data) > 1:
            inc_mean = cluster_data["estimated_income"].mean()
            inc_std = cluster_data["estimated_income"].std()
            inc_cv = (inc_std / inc_mean) * 100 if inc_mean != 0 else 0
            cvs.append(inc_cv)
        else:
            cvs.append(0)
    
    max_cv = max(cvs) if cvs else 0
    return score, max_cv

if __name__ == "__main__":
    print(f"{'Bins':<5} | {'Weight':<7} | {'K':<3} | {'Strat':<10} | {'Silh':<7} | {'Max CV':<7}")
    print("-" * 60)
    
    configs = [
        (13, 50, 6, "kmeans"),
        (13, 100, 6, "kmeans"),
        (20, 50, 6, "kmeans"),
        (20, 100, 6, "kmeans"),
        (13, 100, 13, "kmeans"),
    ]
    
    for b, w, k, s in configs:
        sil, cv = analyze_cv(b, w, k, s)
        print(f"{b:<5} | {w:<7} | {k:<3} | {s:<10} | {sil:.4f} | {cv:.2f}%")
