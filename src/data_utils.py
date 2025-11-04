import pandas as pd
from typing import List
from typing import Literal

DATA_DIR = "../data/"


def load_data(
    file_name: str,
    index_col: str | int | None,
    extension: Literal["csv", "tsv"] = "csv",
) -> pd.DataFrame:
    """Load data from a file into a pandas DataFrame.

    Arguments:
        file_name -- name of the file (without extension)
        index_col -- name of the column to use as the index
        extension -- file extension (default is "csv")

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    separator = "\t" if extension == "tsv" else ","
    if index_col is None:
        return pd.read_csv(f"{DATA_DIR}{file_name}.{extension}", sep=separator)
    return pd.read_csv(
        f"{DATA_DIR}{file_name}.{extension}", sep=separator, index_col=index_col
    )


def concat_dataframes(*args: pd.DataFrame) -> pd.DataFrame:
    """Concat multiple dataframes

    Returns:
        A pandas DataFrame containing the concatenated data.
    """
    return pd.concat(args, axis=1, join="inner")  # type: ignore


def save_data(df: pd.DataFrame, file_name: str) -> None:
    """Save a pandas DataFrame to a CSV file.

    Arguments:
        df -- the pandas DataFrame to save
        file_name -- name of the file (without extension)
    """
    df.to_csv(f"{DATA_DIR}{file_name}.csv")


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Check for missing values in a pandas DataFrame.

    Arguments:
        df -- the pandas DataFrame to check

    Returns:
        A pandas Series containing the count of missing values per column.
    """
    return df.isnull().sum()


def get_column_names_and_object_types(df: pd.DataFrame) -> pd.DataFrame:
    """Get column names and their object types from a pandas DataFrame.

    Arguments:
        df -- the pandas DataFrame to analyze

    Returns:
        A pandas DataFrame containing the column names and their object types.
    """
    return (
        pd.DataFrame(df.dtypes)
        .reset_index()
        .rename(columns={"index": "column_name", 0: "object_type"})
    )


def replace_comas_with_dots(
    df: pd.DataFrame, columns: List[str] | None = None
) -> pd.DataFrame:
    """Replace commas with dots in specified columns of a pandas DataFrame.

    Arguments:
        df -- the pandas DataFrame to modify
        columns -- list of column names to apply the replacement

    Returns:
        A pandas DataFrame with commas replaced by dots in the specified columns.
    """
    if columns is None:
        columns = df.columns.tolist()
    for col in columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".", regex=False).astype(float)
    return df
