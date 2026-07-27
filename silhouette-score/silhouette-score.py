import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    n = X.shape[0]

    sq_norms = np.sum(X**2, axis=1)

    D2 = np.maximum(0, sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(X, X.T))
    D = np.sqrt(D2)

    _, mapped_labels = np.unique(labels, return_inverse=True)
    K = np.max(mapped_labels) + 1

    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), mapped_labels] = 1

    cluster_counts = one_hot.sum(axis=0)

    sum_dist_to_clusters = D @ one_hot

    counts_a = np.maximum(1, cluster_counts[mapped_labels] - 1)
    a = sum_dist_to_clusters[np.arange(n), mapped_labels] / counts_a

    avg_dist_to_clusters = sum_dist_to_clusters / cluster_counts

    avg_dist_to_clusters[np.arange(n), mapped_labels] = np.inf
    b = np.min(avg_dist_to_clusters, axis=1)

    max_ab = np.maximum(a,b)
    s = np.zeros(n)

    mask = max_ab > 0
    s[mask] = (b[mask] - a[mask]) / max_ab[mask]

    s[cluster_counts[mapped_labels] == 1] = 0.0

    return float(np.mean(s))
    