import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN

embeddings = np.load("sample/embeddings.npy")
df = pd.read_csv("sample/cleaned_comments.csv")

# Dimensionality reduction to 5 from 384 dimensions to enhance clustering quality
model = UMAP(n_components=5, random_state=42)
new_embeddings = model.fit_transform(embeddings)

# Autodetection of clusters in dataset
hdb_object = HDBSCAN(min_cluster_size=160)
clusters = hdb_object.fit_predict(new_embeddings)

df['cluster'] = clusters

for cluster_id in sorted(df['cluster'].unique()):
    print(f"\n=== Klaster {cluster_id} (n={sum(clusters == cluster_id)}) ===")
    sample = df[df['cluster'] == cluster_id]['body'].sample(min(5, sum(clusters == cluster_id)), random_state=42)
    for text in sample:
        print(f"- {text[:150]}")