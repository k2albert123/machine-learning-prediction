import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

def test_score(n_bins=15, income_weight=20, n_clusters=6, strategy="kmeans"):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    income = df["estimated_income"].to_numpy().reshape(-1, 1)
    price = df["selling_price"].to_numpy().reshape(-1, 1)
    
    kbd_i = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    kbd_p = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    
    income_b = kbd_i.fit_transform(income)
    price_b = kbd_p.fit_transform(price)
    
    X = np.column_stack([income_b * income_weight, price_b])
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels)
    print(f"Params: bins={n_bins}, weight={income_weight}, clusters={n_clusters}, strat={strategy} -> Silhouette: {score:.4f}")
    return score

if __name__ == "__main__":
    test_score(6, 10, 6, "kmeans")
    test_score(7, 10, 6, "kmeans")
    test_score(6, 5, 6, "kmeans")
    test_score(6, 10, 6, "uniform")
    test_score(7, 10, 6, "uniform")
