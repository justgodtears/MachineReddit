from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd

embeddings = np.load("sample/embeddings.npy")

hdb_object = HDBSCAN(min_cluster_size=100)
clusters_raw = hdb_object.fit_predict(embeddings)
print(pd.Series(clusters_raw).value_counts())

