import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN

embeddings = np.load("sample/embeddings.npy")
df = pd.read_csv("sample/cleaned_comments.csv")

# Dimensionality reduction to 5 from 384 dimensions to enhance clustering quality
model = UMAP(n_components=5, n_neighbors=10, random_state=42)

#print(f"Embeddings: {embeddings.shape}")
#print(f"Dataframe: {df.shape}")

new_embeddings = model.fit_transform(embeddings)
#print(f"Embeddings after dimensionality reduction: {new_embeddings.shape}")

# Autodetection of clusters in dataset
hdb_object = HDBSCAN(min_cluster_size=100)
clusters = hdb_object.fit_predict(new_embeddings)
#print(f"Clusters: {clusters.shape}")
#print(f"Unique clusters: {np.unique(clusters)}")

result = pd.Series(clusters).value_counts()
#print(result)

model_2d = UMAP(n_components=2, random_state=42)
embeddings_2d = model_2d.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab10', s=5)
plt.colorbar()
plt.show()