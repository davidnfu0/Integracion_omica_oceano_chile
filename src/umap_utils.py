import pandas as pd
import umap


def run_umap(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Runs UMAP dimensionality reduction on a DataFrame.

    Arguments:
        df -- Input DataFrame (samples x features).

    Keyword Arguments:
        n_neighbors -- The size of local neighborhood (default: 15).
                       Larger values result in more global views.
        min_dist -- The effective minimum distance between embedded points (default: 0.1).
        n_components -- The dimension of the space to embed into (default: 2).
        metric -- The metric to use to compute distances in high dimensional space (default: 'euclidean').
        random_state -- Seed for reproducibility.

    Returns:
        A DataFrame containing the UMAP embedding.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(df)

    columns = [f"UMAP_{i + 1}" for i in range(n_components)]
    return pd.DataFrame(embedding, index=df.index, columns=columns)
