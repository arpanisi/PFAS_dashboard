from umap import UMAP
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np


def umap_network(data=None, n_components=2, mapper=None, weight_threshold=0.5,
                 n_neighbors=30, metric='euclidean', random_state=42):

    if mapper is None:
        mapper = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=1.0, n_epochs=100,
                       random_state=random_state, metric=metric)
        embedding = mapper.fit_transform(data)
    else:
        embedding = mapper.transform(data)

    emb_graph = nx.from_scipy_sparse_array(mapper.graph_)

    for i, point in enumerate(embedding):
        if n_components == 2:
            emb_graph.add_node(i, position=(point[0], point[1]))
        elif n_components == 3:
            emb_graph.add_node(i, position=(point[0], point[1], point[2]))

    # Remove edges with weight below the threshold
    edges_to_remove = [(u, v) for u, v, w in emb_graph.edges(data='weight') if w < weight_threshold]
    emb_graph.remove_edges_from(edges_to_remove)

    return emb_graph, mapper, embedding


def graph_from_embedding(embedding, weight_threshold=0.5, n_neighbors=30):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    G = nx.Graph()

    n_samples = embedding.shape[0]

    # Add all nodes to the graph first
    for i in range(n_samples):
        G.add_node(i)

    for i in range(n_samples):
        for j in range(1, n_neighbors):  # Skip the first neighbor because it's the node itself
            # Only add edges if the distance is greater than or equal to the threshold
            if distances[i, j] >= weight_threshold:
                G.add_edge(i, indices[i, j], weight=distances[i, j])

    # Remove edges with weight below the threshold
    edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w < weight_threshold]
    G.remove_edges_from(edges_to_remove)

    return G


def compute_topographic_product(X_high_dim, X_low_dim, k=30):
    """
    Computes the Topographic Product between high-dimensional data and low-dimensional embeddings.

    Parameters:
    -----------
    X_high_dim : ndarray of shape (n_samples, n_features)
        High-dimensional data.

    X_low_dim : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding (e.g., from UMAP).

    k : int, optional (default=10)
        The number of nearest neighbors to consider for computing the topographic product.

    Returns:
    --------
    topographic_product : float
        The computed topographic product.
    """

    # Step 1: Compute k-nearest neighbors in high-dimensional space
    nbrs_high = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_high_dim)
    distances_high, indices_high = nbrs_high.kneighbors(X_high_dim)

    # Step 2: Compute k-nearest neighbors in low-dimensional space
    nbrs_low = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_low_dim)
    distances_low, indices_low = nbrs_low.kneighbors(X_low_dim)

    # Step 3: Calculate common neighbors (C_k)
    common_neighbors = []
    for i in range(X_high_dim.shape[0]):
        # Find intersection of neighbors for point i in both high and low dimensions
        common_k = len(set(indices_high[i]).intersection(set(indices_low[i])))
        common_neighbors.append(common_k)

    # Step 4: Compute the Topographic Product
    tp_values = []
    for i in range(1, k):  # Start from 1 because k = 1 means the node itself
        C_k = common_neighbors[i]
        K_k = i + 1  # K_k is just the rank, which is i+1
        tp_values.append(C_k / K_k)

    # Take the log of the product of tp_values and exponentiate to avoid numerical underflow
    log_tp = np.sum(np.log(tp_values))
    topographic_product = np.exp(log_tp)

    return topographic_product


def jaccard_similarity(G1, G2):
    edges_G1 = set(G1.edges())
    edges_G2 = set(G2.edges())
    intersection = edges_G1.intersection(edges_G2)
    union = edges_G1.union(edges_G2)
    return len(intersection) / len(union)


def force_layout(G):

    pos = nx.forceatlas2_layout(G)



