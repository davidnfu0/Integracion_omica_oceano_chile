import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
from typing import Dict, Any
import partipy as pt


def fit_pcha_partipy(
    X: np.ndarray,
    k: int,
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 42,
    delta: float = 0.0,
    centering: bool = True,
    scaling: bool = True,
) -> Dict[str, Any]:
    """
    Envuelve partipy.AA para tener la misma interfaz que fit_pcha clásico.

    Usa Archetypal Analysis (PCHA / Frank-Wolfe) tal como en ParTIpy:
      X ≈ A_mat @ Z
    donde:
      - Z (k x d) son los arquetipos
      - A_mat (N x k) son los coeficientes convexos por muestra.

    Parámetros principales:
        X        : datos (N x d)
        k        : número de arquetipos
        delta    : relajación del constraint convexo sobre los arquetipos
                   (0.0 = dentro del hull, 0.1–0.25 = pueden salir un poco)
        centering, scaling : igual que en partipy.AA
    """
    n, d = X.shape

    # Crear objeto de Archetypal Analysis de ParTIpy
    AA = pt.AA(
        n_archetypes=k,
        init="plus_plus",
        optim="projected_gradients",
        max_iter=max_iter,
        rel_tol=tol,
        early_stopping=True,
        delta=delta,
        centering=centering,
        scaling=scaling,
        seed=seed,
        verbose=False,
    )

    # Ajustar arquetipos
    AA.fit(X)

    # Z = arquetipos (k x d)
    Z = AA.Z

    # A_mat = coeficientes (N x k) para reconstruir X desde Z
    A_mat = AA.transform(X)

    # Reconstrucción en el espacio original
    X_hat = A_mat @ Z

    # RSS y EV en el espacio original (sin depender de cómo centre/escale internamente)
    rss = np.linalg.norm(X - X_hat, "fro") ** 2
    tss = np.sum((X - X.mean(axis=0)) ** 2)
    ev = 1.0 - (rss / tss)

    # t-ratio: volumen politopo / volumen convex hull de los datos (solo d<=3)
    t_ratio = np.nan
    if d <= 3 and k > d:
        try:
            vol_data = ConvexHull(X).volume
            vol_arch = ConvexHull(Z[:, :d]).volume
            t_ratio = vol_arch / vol_data
        except Exception:
            pass

    return {
        "archetypes": Z,
        "S_matrix": A_mat,  # mismo nombre que antes
        "explained_variance": ev,
        "t_ratio": t_ratio,
        "iters": AA.n_iter_ if hasattr(AA, "n_iter_") else max_iter,
    }


def calculate_enrichment(
    data: np.ndarray,
    archetypes: np.ndarray,
    metadata: pd.DataFrame,
    feature_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calcula el enriquecimiento de una característica a medida que nos acercamos a un arquetipo.

    Paper logic: "bin data using optimal bin size and determine which feature is maximal in the bin closest to an archetype"

    Arguments:
        data -- Coordenadas de las muestras (PCAs o Factores).
        archetypes -- Coordenadas de los arquetipos.
        metadata -- DataFrame con los valores de la característica a evaluar.
        feature_col -- Nombre de la columna en metadata.
    """
    results = []
    k = len(archetypes)

    vals = metadata[feature_col].values
    # Manejar NaNs
    mask = ~np.isnan(vals)
    data_clean = data[mask]
    vals_clean = vals[mask]

    for i in range(k):
        # 1. Calcular distancia Euclidiana de cada punto al Arquetipo i
        arch_coord = archetypes[i]
        dists = np.linalg.norm(data_clean - arch_coord, axis=1)

        # 2. Crear dataframe temporal
        df_temp = pd.DataFrame({"dist": dists, "val": vals_clean})

        # 3. Bining por cuantiles de distancia
        try:
            df_temp["bin"] = pd.qcut(
                df_temp["dist"], q=n_bins, labels=False, duplicates="drop"
            )
        except ValueError:
            # Fallback si hay muchos duplicados (poca resolución)
            df_temp["bin"] = pd.cut(df_temp["dist"], bins=n_bins, labels=False)

        # 4. Calcular estadísticas
        bin_means = df_temp.groupby("bin")["val"].mean()

        # 5. Test Mann-Whitney: ¿Es el bin más cercano (0) significativamente mayor que el resto?
        group_close = df_temp[df_temp["bin"] == 0]["val"]
        group_far = df_temp[df_temp["bin"] > 0]["val"]

        try:
            stat, pval = mannwhitneyu(group_close, group_far, alternative="greater")
        except:
            pval = 1.0

        results.append(
            {
                "archetype": i + 1,
                "feature": feature_col,
                "p_value": pval,
                "enrichment_curve": bin_means.values,
                "max_enrichment": bin_means.iloc[0]
                / (df_temp["val"].mean() + 1e-9),  # Fold change vs mean
            }
        )

    return pd.DataFrame(results)
