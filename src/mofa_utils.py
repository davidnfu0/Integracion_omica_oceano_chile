import pandas as pd
import scanpy as sc
import muon as mu

CLR_PROCESSED_DATA_DIR = "../data/clr_data/"


def load_anndata(file_name: str) -> sc.AnnData:
    """Loads an AnnData object from a CSV file located in the CLR_PROCESSED_DATA_DIR.

    Arguments:
        file_name -- Name of the CSV file to load.

    Returns:
        An AnnData object containing the loaded data.
    """
    return sc.AnnData(
        pd.read_csv(f"{CLR_PROCESSED_DATA_DIR}{file_name}.csv", index_col=0)
    )


def create_mudata(
    biogeochemical_genes_file_name: str,
    metabolic_genes_file_name: str,
    Taxa_order_file_name: str,
    Taxa_phylum_file_name: str,
) -> mu.MuData:
    """Creates a MuData object from multiple AnnData objects loaded from specified CSV files.

    Arguments:
        biogeochemical_genes_file_name -- Name of the CSV file containing biogeochemical genes data.
        metabolic_genes_file_name -- Name of the CSV file containing metabolic genes data.
        Taxa_order_file_name -- Name of the CSV file containing Taxa order data.
        Taxa_phylum_file_name -- Name of the CSV file containing Taxa phylum data.

    Returns:
        A MuData object containing the loaded AnnData objects.
    """
    biogeochemical_genes = load_anndata(biogeochemical_genes_file_name)
    metabolic_genes = load_anndata(metabolic_genes_file_name)
    taxa_order = load_anndata(Taxa_order_file_name)
    taxa_phylum = load_anndata(Taxa_phylum_file_name)

    mdata = mu.MuData(
        {
            "biogeochemical_genes": biogeochemical_genes,
            "metabolic_genes": metabolic_genes,
            "taxa_order": taxa_order,
            "taxa_phylum": taxa_phylum,
        }
    )
    return mdata
