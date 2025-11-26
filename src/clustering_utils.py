import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, pairwise_distances
from kneed import KneeLocator


def run_dbscan(data: pd.DataFrame, eps: float, min_samples: int) -> Dict[str, Any]:
    """
    Runs DBSCAN clustering and computes metrics.

    Parameters:
      data (pd.DataFrame): Samples x features.
      eps (float): The maximum distance between two samples for one to be considered
                   as in the neighborhood of the other.
      min_samples (int): The number of samples in a neighborhood for a point to be
                         considered as a core point.
    Returns:
      Dict[str, Any]: {
        model: Fitted DBSCAN model,
        labels: Cluster labels,
        n_clusters: Number of clusters found (excluding noise),
      n_noise: Number of noise points,
      silhouette: Silhouette score (if applicable, else -1),
      }
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    sil = -1.0
    if n_clusters > 1:
        sil = silhouette_score(data, labels)

    return {
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": sil,
    }


def calculate_separation(
    data: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> float:
    """
    Compute between-cluster separation (between-cluster sum of squares, BCSS).

    Parameters:
      data (np.ndarray): Array of shape (n_samples, n_features) with original data.
      labels (np.ndarray): Cluster labels of shape (n_samples,).
      centroids (np.ndarray): Cluster centroids of shape (n_clusters, n_features).

    Returns:
      float: BCSS value measuring separation among clusters (higher means more separated).
    """
    global_mean = np.mean(data, axis=0)
    bcss = 0
    for i, centroid in enumerate(centroids):
        nk = np.sum(labels == i)
        diff = centroid - global_mean
        bcss += nk * np.dot(diff, diff)
    return bcss


def run_kmeans_optimization(
    data: pd.DataFrame, k_range: range = range(2, 11), random_state: int = 42
) -> Dict[str, Any]:
    """
    Runs KMeans across a range of k values, computing optimization metrics.

    Parameters:
      data (pd.DataFrame): Samples x features.
      k_range (range): Candidate numbers of clusters (default range(2, 11)).
      random_state (int): Seed for reproducibility (default 42).

    Metrics per k:
      inertia: Within-cluster sum of squares.
      silhouette: Average silhouette score.
      separation: Between-cluster sum of squares (BCSS).

    Returns:
      Dict[str, Any]: {
      results: {k: {model, labels, inertia, silhouette, separation}},
      optimal_k: Elbow k from KneeLocator (fallback last k),
      inertias: List[float],
      silhouettes: List[float],
      k_range: List[int]
      }
    """
    inertias = []
    silhouettes = []
    models = {}

    X = data.values

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)

        models[k] = {
            "model": model,
            "labels": labels,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels),
            "separation": calculate_separation(X, labels, model.cluster_centers_),
        }
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    kneedle = KneeLocator(
        list(k_range), inertias, curve="convex", direction="decreasing"
    )
    optimal_k = kneedle.elbow if kneedle.elbow else k_range[-1]

    return {
        "results": models,
        "optimal_k": optimal_k,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_range": list(k_range),
    }


def get_proximity_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Euclidean pairwise proximity (distance) matrix.

    Arguments:
      data (pd.DataFrame): Samples x features.

    Returns:
      pd.DataFrame: Square (n_samples x n_samples) matrix of Euclidean distances
      with both index and columns matching data.index.
    """
    dist_matrix = pairwise_distances(data, metric="euclidean")
    return pd.DataFrame(dist_matrix, index=data.index, columns=data.index)


# ... (Mantén tus funciones anteriores de K-Means y calculate_separation) ...


def optimize_gmm(
    data: pd.DataFrame, k_range: range = range(2, 11), random_state: int = 42
) -> Dict[str, Any]:
    """
    Runs GMM for a range of components and returns BIC/AIC metrics.
    """
    bics = []
    aics = []
    silhouettes = []
    models = {}

    X = data.values

    for k in k_range:
        model = GaussianMixture(n_components=k, random_state=random_state)
        model.fit(X)
        labels = model.predict(X)

        models[k] = {
            "model": model,
            "labels": labels,
            "bic": model.bic(X),
            "aic": model.aic(X),
            "silhouette": silhouette_score(X, labels),
        }
        bics.append(model.bic(X))
        aics.append(model.aic(X))
        silhouettes.append(silhouette_score(X, labels))

    return {
        "results": models,
        "bics": bics,
        "aics": aics,
        "silhouettes": silhouettes,
        "k_range": list(k_range),
    }


def calculate_k_distance(data: pd.DataFrame, k: int = 5) -> np.ndarray:
    """
    Calculates the k-nearest neighbor distances for DBSCAN parameter tuning (eps).
    Returns the distances sorted and the indices.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    # Sort distance values by the k-th neighbor (column index k-1)
    sorted_distances = np.sort(distances[:, k - 1], axis=0)
    return sorted_distances


def grid_search_dbscan(
    data: pd.DataFrame, eps_range: List[float], min_samples_range: List[int]
) -> pd.DataFrame:
    """
    Runs a Grid Search for DBSCAN and returns a summary DataFrame with metrics.
    """
    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)

            # Metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)

            sil = -1.0
            if n_clusters > 1:
                sil = silhouette_score(data, labels)

            results.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_ratio": round(noise_ratio, 3),
                    "silhouette": round(sil, 3),
                    "labels": labels,  # Guardamos etiquetas temporalmente
                }
            )

    return pd.DataFrame(results)
