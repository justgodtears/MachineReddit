import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
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

# Saving clustering result in new columns
df['cluster'] = clusters

# Skipping noise (-1)
cluster_ids = sorted([c for c in df['cluster'].unique() if c != -1])

# Joining all comments in one document in every cluster
documents = []
for cluster_id in cluster_ids:
    cluster_text = " ".join(df[df['cluster'] == cluster_id]['body'])
    documents.append(cluster_text)

# Custom stop words list for filtering artifacts and common words
custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    'like', 'just', 'don', 'really', 'gt', 'com', 'http', 'https', 'll', 've', 'www', 'php', 'pitjet'
]

# Picking up only 1000 most important words for clarity and efficiency
vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=1000)
tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

# Sorts results, picks up last 10 and reverses it, so the biggest one is first
for i, cluster_id in enumerate(cluster_ids):
    scores = tfidf_matrix[i].toarray().flatten()
    top_indices = scores.argsort()[-10:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"Cluster {cluster_id}: {top_words}")
