from typing import Tuple
import numpy as np
import pandas as pd
from skbio.stats.composition import multi_replace, clr, closure


def filter_low_variance_and_sparse(
    df: pd.DataFrame,
    min_prevalence: float = 1e-6,
    min_variance: float = 1e-8,
) -> pd.DataFrame:
    """Filter features in the DataFrame based on minimum prevalence and variance.

    Arguments:
        df -- The input DataFrame to filter.

    Keyword Arguments:
        min_prevalence -- The minimum prevalence threshold for features (default: {1e-6})
        min_variance -- The minimum variance threshold for features (default: {1e-8})

    Returns:
        A filtered DataFrame containing only the features that meet the specified thresholds.
    """
    prevalence = (df > 0).sum(axis=0) / df.shape[0]
    keep_prevalence = prevalence >= min_prevalence

    var = df.var(axis=0)
    keep_var = var >= min_variance

    keep = keep_prevalence & keep_var
    df_filtered = df.loc[:, keep]

    return df_filtered


def handle_zeros_and_closure(
    df: pd.DataFrame, min_prevalence: float = 1e-6, min_variance: float = 1e-8
) -> pd.DataFrame:
    """Handle zeros in the DataFrame using multiplicative replacement and apply closure transformation.

    Arguments:
        df -- The input DataFrame to process.

    Keyword Arguments:
        min_prevalence -- The minimum prevalence threshold for features (default: {1e-6})
        min_variance -- The minimum variance threshold for features (default: {1e-8})

    Returns:
        A DataFrame with zeros handled and closure transformation applied.
    """
    df_filtered = filter_low_variance_and_sparse(df, min_prevalence, min_variance)
    X = df_filtered.to_numpy(float)
    X = closure(np.maximum(X, 0.0))
    X_nozero = multi_replace(X)

    return pd.DataFrame(X_nozero, index=df_filtered.index, columns=df_filtered.columns)


def clr_transform(
    df: pd.DataFrame,
    min_prevalence: float = 1e-6,
    min_variance: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply CLR transformation to the DataFrame after handling zeros and filtering.

    Arguments:
        df -- The input DataFrame to process.

    Keyword Arguments:
        min_prevalence -- The minimum prevalence threshold for features (default: {1e-6})
        min_variance -- The minimum variance threshold for features (default: {1e-8})

    Returns:
        A tuple containing the CLR transformed DataFrame and the filtered DataFrame.
    """
    df_comp = handle_zeros_and_closure(df, min_prevalence, min_variance)
    df_clr = pd.DataFrame(
        clr(df_comp.to_numpy(float)), index=df_comp.index, columns=df_comp.columns
    )
    assert np.allclose(df_clr.mean(axis=1).to_numpy(), 0.0, atol=1e-9)
    return df_clr, df_comp
