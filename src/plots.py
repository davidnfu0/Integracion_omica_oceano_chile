import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # for typing 3D axes
from scipy.spatial import ConvexHull


def scatter_plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "viridis",
    s: int = 50,
    alpha: float = 0.7,
    filename: str | None = None,
    colorbar_label: str = "",
) -> None:
    """
    Plots a 2D scatter plot.

    Arguments:
        x -- vector of x coordinates
        y -- vector of y coordinates
        c -- vector of colors or values for coloring
        xlabel -- label for the x-axis
        ylabel -- label for the y-axis
        title -- title of the plot
    """
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        alpha=alpha,
        edgecolors="w",
        s=s,
    )
    cbar = plt.colorbar(scatter)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight") if filename else None
    plt.show()


def scatter_plot_3d(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "viridis",
    s: int = 100,
    alpha: float = 0.7,
    filename: str | None = None,
) -> None:
    """
    Plots a scatter plot with the given parameters.

    Arguments:
        x -- vector of x coordinates
        y -- vector of y coordinates
        c -- vector of colors or values for coloring
        xlabel -- label for the x-axis
        ylabel -- label for the y-axis
        title -- title of the plot
    Keyword Arguments:
        cmap -- color map to use for coloring (default: {"viridis"})
        s -- marker size (default: {100})
        alpha -- marker transparency (default: {0.7})
        filename -- filename to save the plot (default: {None})
    """
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        alpha=alpha,
        edgecolors="w",
        s=s,
    )
    plt.colorbar(scatter)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches="tight") if filename else None
    plt.show()


def doble_scatter_plot_3d(
    x1: np.ndarray,
    y1: np.ndarray,
    c1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    c2: np.ndarray,
    xlabel1: str,
    ylabel1: str,
    xlabel2: str,
    ylabel2: str,
    title1: str,
    title2: str,
    cmap: str = "viridis",
    s: int = 100,
    alpha: float = 0.7,
    filename: str | None = None,
) -> None:
    """Create two side-by-side scatter plots for comparison.

    Arguments:
        x1 -- 1D array of x coordinates for first plot
        y1 -- 1D array of y coordinates for first plot
        c1 -- 1D array of values mapped to colors for first plot
        x2 -- 1D array of x coordinates for second plot
        y2 -- 1D array of y coordinates for second plot
        c2 -- 1D array of values mapped to colors for second plot
        xlabel1 -- x-axis label for first plot
        ylabel1 -- y-axis label for first plot
        xlabel2 -- x-axis label for second plot
        ylabel2 -- y-axis label for second plot
        title1 -- title for first subplot
        title2 -- title for second subplot (variable name kept as given)

    Keyword Arguments:
        cmap -- colormap name (default: "viridis")
        s -- marker size (default: 100)
        alpha -- marker transparency (default: 0.7)
        filename -- path to save figure, if provided (default: None)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=False)

    sc1 = axes[0].scatter(x1, y1, c=c1, cmap=cmap, alpha=alpha, edgecolors="w", s=s)
    axes[0].set_xlabel(xlabel1)
    axes[0].set_ylabel(ylabel1)
    axes[0].set_title(title1)
    fig.colorbar(sc1, ax=axes[0])

    sc2 = axes[1].scatter(x2, y2, c=c2, cmap=cmap, alpha=alpha, edgecolors="w", s=s)
    axes[1].set_xlabel(xlabel2)
    axes[1].set_ylabel(ylabel2)
    axes[1].set_title(title2)
    fig.colorbar(sc2, ax=axes[1])

    fig.suptitle("Comparison of Two Scatter Plots", fontsize=16)
    for ax in axes:
        ax.grid(True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_variance_explained(variance_explained) -> None:
    """
    Plots the variance explained per factor and view as a heatmap.

    Arguments:
        variance_explained -- DataFrame with variance explained values
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(variance_explained, aspect="auto", cmap="Blues")
    plt.xticks(
        ticks=range(len(variance_explained.columns)),
        labels=variance_explained.columns,
        rotation=45,
    )
    plt.yticks(
        ticks=range(len(variance_explained.index)),
        labels=variance_explained.index,
    )
    plt.colorbar(label="R2 (%)")
    plt.title("Variance Explained per Factor and View")
    plt.tight_layout()
    plt.show()


def plot_factor_top_features(
    df_W: "pd.DataFrame", num_factors: int, num_top_features: int
) -> list["pd.DataFrame"]:
    """
    Plots the weights of features for each factor and highlights the top features.

    Arguments:
        df_W -- DataFrame containing feature weights for each factor
        num_factors -- number of factors to plot
        num_top_features -- number of top features to highlight per factor

    Returns:
        A list of DataFrames, each containing the top features for a factor.
    """
    fig, axs = plt.subplots(num_factors, 1, figsize=(18, 3 * num_factors), sharex=True)
    feature_tables = []

    for i in range(num_factors):
        factor = str(i)
        weights = df_W[factor]
        axs[i].plot(weights.values, label=factor, alpha=0.7, linewidth=2)
        top_idx = weights.abs().nlargest(num_top_features).index
        top_pos = [weights.index.get_loc(idx) for idx in top_idx]
        axs[i].scatter(
            top_pos, weights.loc[top_idx].values, color="red", s=50, zorder=3
        )
        for pos, idx in zip(top_pos, top_idx):
            axs[i].text(
                pos,
                weights.loc[idx],
                str(pos),
                color="black",
                fontsize=12,
                ha="right",
                va="bottom",
                rotation=45,
            )
        feature_tables.append(
            pd.DataFrame({"Feature_number": top_pos, "Feature_name": top_idx})
        )
        axs[i].set_title(f"Features weights in biogeochemical_genes - {factor}")
        axs[i].set_ylabel("Weight")
        axs[i].legend(loc="upper right")
        axs[i].set_xticks([])
        axs[i].set_ylim(
            weights.min() * (1.8),
            weights.max() * (1.8),
        )
        axs[i].grid()
    plt.tight_layout()
    plt.show()

    return feature_tables


def plot_factor_metadata(
    mofa_1: np.ndarray,
    mofa_2: np.ndarray,
    metadata: pd.DataFrame,
    metadata_cols: list[str],
    s: int = 50,
    alpha: float = 0.7,
) -> None:
    """
    Plots scatter plots of factors colored by metadata columns.

    Arguments:
        mofa_1 -- 1D array of factor values (samples x factor)
        mofa_2 -- 1D array of factor values (samples x factor)
        metadata -- DataFrame containing metadata values
        metadata_cols -- List of metadata column names to use for coloring
        s -- marker size (default: 50)
        alpha -- marker transparency (default: 0.7)
    """
    num_cols = len(metadata_cols)
    graph_rows = num_cols // 3
    if num_cols / 3 > num_cols // 3:
        graph_rows += 1
    fig, axes = plt.subplots(graph_rows, 3, figsize=(15, 5 * graph_rows))
    axes = axes.flatten()

    for i, col in enumerate(metadata_cols):
        ax = axes[i]
        scatter = ax.scatter(
            mofa_1,
            mofa_2,
            c=metadata[col],
            cmap="viridis",
            alpha=alpha,
            edgecolors="w",
            s=s,
        )
        ax.set_title(f"Metadata: {col}")
        ax.set_xlabel("Factor 1")
        ax.set_ylabel("Factor 2")
        axes[i].grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=ax, label=col)
    plt.tight_layout()
    plt.show()


def plot_dim_reduction_metadata(
    emb_x: np.ndarray,
    emb_y: np.ndarray,
    metadata: pd.DataFrame,
    metadata_cols: list[str],
    dim_name: str = "UMAP",
    s: int = 40,
) -> None:
    """
    Plots dimensionality reduction results colored by multiple metadata columns.

    Arguments:
        emb_x -- x coordinates of the embedding
        emb_y -- y coordinates of the embedding
        metadata -- DataFrame containing metadata values
        metadata_cols -- List of metadata column names to use for coloring

    Keyword Arguments:
        dim_name -- Name of the dimensionality reduction method (default: {"UMAP"})
        s -- Marker size (default: {40})
    """
    num_cols = len(metadata_cols)
    graph_rows = (num_cols + 2) // 3

    fig, axes = plt.subplots(graph_rows, 3, figsize=(18, 5 * graph_rows))
    axes = axes.flatten()

    for i, col in enumerate(metadata_cols):
        ax = axes[i]

        c_values = metadata[col]
        if c_values.dtype == "object":
            c_values = pd.factorize(c_values)[0]

        scatter = ax.scatter(
            emb_x,
            emb_y,
            c=c_values,
            cmap="viridis",
            alpha=0.7,
            edgecolors="w",
            s=s,
        )
        ax.set_title(f"{col}")
        ax.set_xlabel(f"{dim_name} 1")
        ax.set_ylabel(f"{dim_name} 2")
        ax.grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=ax, label=col)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_elbow_silhouette(
    k_range: list, inertias: list, silhouettes: list, optimal_k: int, title_prefix: str
):
    """Plots Inertia (Elbow) and Silhouette scores side by side."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow
    ax[0].plot(k_range, inertias, "bo-", markerfacecolor="red")
    ax[0].axvline(
        x=optimal_k, color="k", linestyle="--", label=f"Optimal k={optimal_k}"
    )
    ax[0].set_title(f"{title_prefix} - Elbow Method (Cohesion)")
    ax[0].set_xlabel("Number of Clusters (k)")
    ax[0].set_ylabel("Inertia")
    ax[0].legend()
    ax[0].grid(True)

    # Silhouette
    ax[1].plot(k_range, silhouettes, "go-", markerfacecolor="orange")
    ax[1].axvline(
        x=optimal_k, color="k", linestyle="--", label=f"Optimal k={optimal_k}"
    )
    ax[1].set_title(f"{title_prefix} - Silhouette Score")
    ax[1].set_xlabel("Number of Clusters (k)")
    ax[1].set_ylabel("Silhouette Score")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_geo_clusters(
    metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    lat_col: str,
    lon_col: str,
    title: str,
    n_clusters: int,
    palette: list | None = None,
    chile_normalize: bool = True,
) -> None:
    """
    Plot distribución geográfica de muestras por cluster con relación 1:1 en ejes.
    """
    if palette is None:
        if n_clusters <= 10:
            palette = sns.color_palette("tab10", 10)[:n_clusters]
        else:
            palette = sns.color_palette("hls", n_clusters)
    cmap = ListedColormap(palette[:n_clusters])

    lon_min = metadata[lon_col].min()
    lon_max = metadata[lon_col].max()
    lat_min = metadata[lat_col].min()
    lat_max = metadata[lat_col].max()

    fig, ax = plt.subplots(figsize=(6, 10))
    scatter = ax.scatter(
        metadata[lon_col],
        metadata[lat_col],
        c=cluster_labels,
        cmap=cmap,
        vmin=0,
        vmax=n_clusters - 1,
        s=30,
        edgecolors="k",
        alpha=0.85,
        linewidths=0.5,
    )

    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters), pad=0.01, shrink=0.8)
    cbar.set_label("Cluster")

    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    if chile_normalize:
        lon_min = -75.0
        lon_max = -66.0
        lat_min = -56.0
        lat_max = -17.0

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_aspect("equal", adjustable="box")

    xticks = np.arange(np.floor(lon_min) - 3, np.ceil(lon_max) + 3, 2)
    yticks = np.arange(np.floor(lat_min) - 3, np.ceil(lat_max) + 3, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()


def plot_geo_clusters_3d(
    metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    lat_col: str,
    lon_col: str,
    depth_col: str,
    title: str,
    n_clusters: int,
    palette: list | None = None,
    chile_normalize: bool = True,
    elev: int = 30,
    azim: int = -60,
) -> None:
    """
    Plot 3D (Longitud, Latitud, Profundidad) con una sola vista.
    Profundidad se muestra negativa (más profundo más negativo).
    """
    if palette is None:
        palette = (
            sns.color_palette("tab10", 10)[:n_clusters]
            if n_clusters <= 10
            else sns.color_palette("hls", n_clusters)
        )
    cmap = ListedColormap(palette[:n_clusters])

    lon_min = metadata[lon_col].min()
    lon_max = metadata[lon_col].max()
    lat_min = metadata[lat_col].min()
    lat_max = metadata[lat_col].max()
    z_vals = -metadata[depth_col].values  # invertir signo
    z_min = z_vals.min()

    if chile_normalize:
        lon_min, lon_max = -75.0, -66.0
        lat_min, lat_max = -56.0, -17.0

    fig = plt.figure(figsize=(10, 7))
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection="3d")

    sc = ax.scatter(
        metadata[lon_col],
        metadata[lat_col],
        z_vals,
        c=cluster_labels,
        cmap=cmap,
        vmin=0,
        vmax=n_clusters - 1,
        s=10,
        edgecolors="k",
        alpha=0.9,
        linewidths=0.4,
    )
    ax.set_title(f"{title}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_zlabel("Profundidad (-m)")

    ax.set_xlim(lon_min - 2, lon_max + 2)
    ax.set_ylim(lat_min - 2, lat_max + 2)
    ax.set_zlim(z_min, 0)
    ax.set_box_aspect((1, 1, 0.4))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle="--", alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.05, shrink=0.6, ticks=range(n_clusters))
    cbar.set_label("Cluster")

    fig.tight_layout()
    plt.show()


def plot_gmm_optimization(k_range: list, bics: list, aics: list, silhouettes: list):
    """Plots BIC/AIC (lower is better) and Silhouette for GMM."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # BIC/AIC
    ax[0].plot(k_range, bics, "b-o", label="BIC")
    ax[0].plot(k_range, aics, "g--x", label="AIC")
    ax[0].set_title("GMM Model Selection (Lower is better)")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_ylabel("Score")
    ax[0].legend()
    ax[0].grid(True)

    # Silhouette
    ax[1].plot(k_range, silhouettes, "r-o")
    ax[1].set_title("GMM Silhouette Score")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_ylabel("Silhouette")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_k_distance_curve(distances: np.ndarray, k: int):
    """Plots the sorted k-distance graph for DBSCAN eps selection."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f"K-Distance Graph (k={k})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-NN Distance (Epsilon candidate)")
    plt.grid(True)
    plt.show()


def plot_metadata_by_cluster(
    metadata: pd.DataFrame, cluster_labels: np.ndarray, numeric_cols: list
):
    """Creates boxplots for metadata variables grouped by cluster."""
    df_plot = metadata.copy()
    df_plot["Cluster"] = cluster_labels

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x="Cluster", y=col, data=df_plot, ax=axes[i], palette="Set2")
        axes[i].set_title(col)
        axes[i].grid(True, alpha=0.3)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ... (imports existentes)


def plot_pareto_ev_curve(results_list: list):
    """Grafica la curva de codo para seleccionar k (número de arquetipos)."""
    ks = [r["k"] for r in results_list]
    evs = [r["ev"] for r in results_list]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, evs, "o-", color="purple", linewidth=2)
    plt.xlabel("Número de Arquetipos (k)")
    plt.ylabel("Varianza Explicada (EV)")
    plt.title("Selección de k (Método del Codo)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_polytope_3d(
    data: np.ndarray,
    archetypes: np.ndarray,
    labels: np.ndarray = None,
    title: str = "Espacio de Pareto",
):
    """
    Visualiza los datos y el politopo (Hull de arquetipos) en 3D.
    """
    if data.shape[1] < 3:
        print("Se requieren al menos 3 dimensiones.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Puntos de datos
    c_vals = labels if labels is not None else "gray"
    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2], c=c_vals, cmap="viridis", s=10, alpha=0.7
    )

    # Arquetipos
    ax.scatter(
        archetypes[:, 0],
        archetypes[:, 1],
        archetypes[:, 2],
        c="red",
        s=50,
        marker="*",
        label="Arquetipos",
        depthshade=False,
    )

    # Dibujar aristas del politopo
    if len(archetypes) >= 4:
        hull = ConvexHull(archetypes[:, :3])
        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(
                archetypes[simplex, 0],
                archetypes[simplex, 1],
                archetypes[simplex, 2],
                "k-",
                lw=1,
            )
    elif len(archetypes) == 3:
        # Triángulo simple
        idx = [0, 1, 2, 0]
        ax.plot(archetypes[idx, 0], archetypes[idx, 1], archetypes[idx, 2], "k-", lw=1)

    ax.set_xlabel("PC/Factor 1")
    ax.set_ylabel("PC/Factor 2")
    ax.set_zlabel("PC/Factor 3")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_enrichment_curves(enrichment_df: pd.DataFrame):
    """Grafica las curvas de enriquecimiento para cada arquetipo."""
    k = len(enrichment_df)
    feature_name = enrichment_df.iloc[0]["feature"]

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]

    for i, row in enrichment_df.iterrows():
        curve = row["enrichment_curve"]
        ax = axes[i]
        x = np.linspace(0, 1, len(curve))

        # Normalizar curva para visualización (relativo al min)
        norm_curve = curve / (np.mean(curve) + 1e-9)

        ax.plot(x, norm_curve, "o-", lw=2, color="tab:blue")
        ax.set_title(f"Arquetipo {row['archetype']}\np={row['p_value']:.1e}")
        ax.set_xlabel("Distancia (0=Cerca)")
        if i == 0:
            ax.set_ylabel("Enriquecimiento Relativo")
        ax.axhline(1.0, ls="--", c="gray")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Enriquecimiento de: {feature_name}", y=1.05)
    plt.tight_layout()
    plt.show()


def plot_regression_results(
    y_true: np.ndarray, y_pred: np.ndarray, target_name: str, r2: float
) -> None:
    """
    Plotea Valores Reales vs Predichos para evaluar la regresión.
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="w", color="teal")

    # Línea de identidad (Predicción Perfecta)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")

    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Prediction of {target_name}\n$R^2 = {r2:.3f}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_factor_importance_bar(
    df_importance: pd.DataFrame, target_name: str, top_n: int = 10
) -> None:
    """Grafica qué Factores son más importantes para la predicción."""
    plt.figure(figsize=(8, 5))
    subset = df_importance.head(top_n)

    # Invertir para que el más importante quede arriba
    subset = subset.iloc[::-1]

    plt.barh(subset["Factor"], subset["Importance"], color="steelblue")
    plt.xlabel("Importance (Gini)")
    plt.title(f"Top Factors driving '{target_name}'")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray, classes: list, title: str, cmap: str = "Blues"
) -> None:
    """Plotea una matriz de confusión con etiquetas."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
