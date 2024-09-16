import numpy as np
from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering(distances: np.array, n_clusters: int) -> np.array:
    """
    Perform hierarchical clustering on a distance matrix.

    Parameters:
    distances (array-like): A distance matrix or array-like object representing pairwise distances between samples.
    n_clusters (int): The number of clusters to form.

    Returns:
    labels (array-like): An array of cluster labels assigned to each sample.

    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    labels = clustering.fit_predict(distances)
    
    return labels