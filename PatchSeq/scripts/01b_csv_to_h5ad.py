#!/usr/bin/env python
"""
01b_csv_to_h5ad.py
Convert exported CSV expression data to h5ad format for Geneformer tokenization.

The source RData files contain log2(CPM+1) values, NOT raw counts.
Since raw counts are unrecoverable (library sizes are lost in CPM normalization),
we store CPM values (2^x - 1) in .X and set n_counts = 1e6 for all cells.

This works correctly with Geneformer's tokenizer because:
  1. The tokenizer normalizes: X / n_counts * target_sum = CPM / 1e6 * 10000
  2. Then divides by gene median factors and RANKS genes by expression
  3. The ranking is preserved regardless of the constant scaling factor

Requirements for Geneformer V1 tokenization:
  - .var must have 'ensembl_id' column with human Ensembl gene IDs
  - .obs must have 'n_counts' (total read counts per cell)
  - .obs should have 'specimen_id' (integer cell ID matching ephys data)

Gene symbols are mapped to Ensembl IDs using Geneformer's V1 gene_name_id_dict.
Genes without a valid mapping are dropped.

Usage:
    conda run -n geneformer python PatchSeq/scripts/01b_csv_to_h5ad.py
"""

import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "PatchSeq", "data")

# Geneformer V1 gene dictionaries
GENE_NAME_ID_DICT = os.path.join(
    PROJECT_ROOT, "geneformer", "gene_dictionaries_30m", "gene_name_id_dict_gc30M.pkl"
)
TOKEN_DICT = os.path.join(
    PROJECT_ROOT, "geneformer", "gene_dictionaries_30m", "token_dictionary_gc30M.pkl"
)


def load_gene_mappings():
    """Load Geneformer V1 gene name -> Ensembl ID mapping and token dictionary."""
    with open(GENE_NAME_ID_DICT, "rb") as f:
        gene_name_id = pickle.load(f)

    with open(TOKEN_DICT, "rb") as f:
        token_dict = pickle.load(f)

    # Only keep Ensembl IDs that are in the token dictionary (model vocabulary)
    token_ensembl_ids = {k for k in token_dict.keys() if k.startswith("ENSG")}

    print(f"Gene name -> Ensembl ID mappings: {len(gene_name_id)}")
    print(f"Ensembl IDs in V1 token dict: {len(token_ensembl_ids)}")

    return gene_name_id, token_ensembl_ids


def build_h5ad(
    expression_csv_gz: str,
    metadata_csv: str,
    gene_name_id: dict,
    token_ensembl_ids: set,
    output_path: str,
    dataset_name: str,
):
    """
    Build an h5ad file from exported CSV data.

    Parameters
    ----------
    expression_csv_gz : str
        Path to gzipped CSV with expression matrix (genes x cells, first col is gene symbol).
    metadata_csv : str
        Path to CSV with cell metadata (must have sample_id, specimen_id columns).
    gene_name_id : dict
        Gene symbol -> Ensembl ID mapping.
    token_ensembl_ids : set
        Set of Ensembl IDs in the model's token dictionary.
    output_path : str
        Path to write the output h5ad file.
    dataset_name : str
        Name for logging.
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Load expression matrix (genes x cells)
    # -----------------------------------------------------------------------
    print(f"Reading expression matrix from {os.path.basename(expression_csv_gz)}...")
    expr_df = pd.read_csv(expression_csv_gz, index_col=0)
    print(f"  Raw matrix shape: {expr_df.shape[0]} genes x {expr_df.shape[1]} cells")

    # Gene symbols are row indices
    gene_symbols = expr_df.index.tolist()

    # -----------------------------------------------------------------------
    # Convert from log2(CPM+1) to CPM
    # -----------------------------------------------------------------------
    # The source RData files store log2(CPM+1) values.
    # Verify: each column should sum to exactly 1e6 after 2^x - 1 transform.
    test_col = expr_df.iloc[:, 0].values
    cpm_test_sum = np.sum(np.power(2.0, test_col) - 1.0)
    assert abs(cpm_test_sum - 1e6) < 1.0, (
        f"Expected log2(CPM+1) data (col sum of 2^x-1 ~ 1e6), got {cpm_test_sum:.1f}"
    )
    print(f"  Confirmed log2(CPM+1) format (inverse-transform col sum = {cpm_test_sum:.0f})")
    print("  Converting to CPM: 2^x - 1")

    # -----------------------------------------------------------------------
    # Map gene symbols to Ensembl IDs
    # -----------------------------------------------------------------------
    print("Mapping gene symbols to Ensembl IDs...")
    ensembl_ids = []
    keep_mask = []
    for gs in gene_symbols:
        eid = gene_name_id.get(gs, None)
        if eid is not None and eid in token_ensembl_ids:
            ensembl_ids.append(eid)
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    keep_mask = np.array(keep_mask)
    n_mapped = keep_mask.sum()
    n_total = len(gene_symbols)
    print(f"  Genes mapped to V1 token dict: {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%)")
    print(f"  Genes dropped (no mapping or not in token dict): {n_total - n_mapped}")

    # Check for duplicate Ensembl IDs (multiple gene symbols -> same Ensembl ID)
    if len(set(ensembl_ids)) < len(ensembl_ids):
        eid_counts = pd.Series(ensembl_ids).value_counts()
        dups = eid_counts[eid_counts > 1]
        print(f"  WARNING: {len(dups)} duplicate Ensembl IDs found. Summing counts for duplicates.")

    # Filter expression matrix to mapped genes only and convert to CPM
    expr_filtered_log2 = expr_df.iloc[keep_mask, :]
    # Convert log2(CPM+1) -> CPM
    expr_filtered = np.power(2.0, expr_filtered_log2.values) - 1.0
    gene_symbols_filtered = [gs for gs, k in zip(gene_symbols, keep_mask) if k]

    # -----------------------------------------------------------------------
    # Build var DataFrame (gene metadata)
    # -----------------------------------------------------------------------
    var_df = pd.DataFrame(
        {
            "gene_symbol": gene_symbols_filtered,
            "ensembl_id": ensembl_ids,
        }
    )

    # Handle duplicate Ensembl IDs by summing CPM values
    if var_df["ensembl_id"].duplicated().any():
        print("  Aggregating duplicate Ensembl IDs by summing CPM values...")
        # expr_filtered is genes x cells numpy array; transpose to cells x genes
        expr_t_df = pd.DataFrame(expr_filtered.T, columns=ensembl_ids)
        expr_agg = expr_t_df.T.groupby(level=0).sum().T
        # Rebuild var
        unique_eids = expr_agg.columns.tolist()
        # For gene_symbol, take the first symbol mapping to each Ensembl ID
        eid_to_symbol = {}
        for gs, eid in zip(gene_symbols_filtered, ensembl_ids):
            if eid not in eid_to_symbol:
                eid_to_symbol[eid] = gs
        var_df = pd.DataFrame(
            {
                "ensembl_id": unique_eids,
                "gene_symbol": [eid_to_symbol[eid] for eid in unique_eids],
            }
        )
        expr_matrix = expr_agg.values  # cells x genes, dense
    else:
        # Transpose: cells x genes (expr_filtered is genes x cells numpy array)
        expr_matrix = expr_filtered.T  # cells x genes

    var_df.index = var_df["ensembl_id"]
    var_df.index.name = None

    print(f"  Final gene count: {var_df.shape[0]}")

    # -----------------------------------------------------------------------
    # Load cell metadata
    # -----------------------------------------------------------------------
    print(f"Reading cell metadata from {os.path.basename(metadata_csv)}...")
    meta_df = pd.read_csv(metadata_csv)
    meta_df.index = meta_df["sample_id"]
    meta_df.index.name = None

    # Ensure cell order matches expression matrix
    cell_ids = expr_df.columns.tolist()
    assert all(
        c in meta_df.index for c in cell_ids
    ), "Some expression cell IDs not found in metadata!"
    meta_df = meta_df.loc[cell_ids]

    # Ensure specimen_id is integer
    meta_df["specimen_id"] = meta_df["specimen_id"].astype(int)

    print(f"  Cells: {meta_df.shape[0]}, Metadata columns: {meta_df.shape[1]}")

    # -----------------------------------------------------------------------
    # Set n_counts
    # -----------------------------------------------------------------------
    # Since .X contains CPM values (not raw counts), set n_counts = 1e6 for
    # all cells. The tokenizer computes X / n_counts * target_sum, so this
    # correctly scales CPM -> target_sum per cell while preserving gene rankings.
    # Note: CPM row sums over the mapped gene subset will be < 1e6 since many
    # genes were dropped. Setting n_counts = 1e6 (the original normalization
    # denominator) ensures the relative expression levels are correct.
    n_counts_all_genes = 1_000_000.0
    meta_df["n_counts"] = n_counts_all_genes

    # Also store the actual CPM sum over mapped genes for reference
    cpm_row_sums = np.array(expr_matrix.sum(axis=1), dtype=np.float64).flatten()
    meta_df["cpm_sum_mapped_genes"] = cpm_row_sums
    print(f"  n_counts set to {n_counts_all_genes:.0f} (CPM convention)")
    print(f"  CPM sum over mapped genes: min={cpm_row_sums.min():.0f}, "
          f"median={np.median(cpm_row_sums):.0f}, max={cpm_row_sums.max():.0f}")

    # -----------------------------------------------------------------------
    # Build AnnData object
    # -----------------------------------------------------------------------
    # Store as sparse matrix for efficiency
    X_sparse = sparse.csr_matrix(expr_matrix.astype(np.float32))

    adata = ad.AnnData(
        X=X_sparse,
        obs=meta_df,
        var=var_df,
    )

    print(f"  AnnData shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  var columns: {list(adata.var.columns)}")

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------
    assert "ensembl_id" in adata.var.columns, "Missing 'ensembl_id' in var!"
    assert "n_counts" in adata.obs.columns, "Missing 'n_counts' in obs!"
    assert "specimen_id" in adata.obs.columns, "Missing 'specimen_id' in obs!"
    assert adata.X.min() >= 0, "Negative values in expression matrix!"

    # Check Ensembl ID format
    sample_eids = adata.var["ensembl_id"].head(5).tolist()
    assert all(
        eid.startswith("ENSG") for eid in sample_eids
    ), f"Invalid Ensembl IDs: {sample_eids}"

    # Verify n_counts is 1e6 (CPM convention)
    assert (adata.obs["n_counts"] == 1_000_000.0).all(), "n_counts should be 1e6 for CPM data!"

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"  Saving to {output_path}...")
    adata.write_h5ad(output_path)
    file_size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved ({file_size_mb:.1f} MB)")

    return adata


def main():
    gene_name_id, token_ensembl_ids = load_gene_mappings()

    # -------------------------------------------------------------------
    # GABA dataset
    # -------------------------------------------------------------------
    gaba_adata = build_h5ad(
        expression_csv_gz=os.path.join(DATA_DIR, "gaba_expression.csv.gz"),
        metadata_csv=os.path.join(DATA_DIR, "gaba_cell_metadata.csv"),
        gene_name_id=gene_name_id,
        token_ensembl_ids=token_ensembl_ids,
        output_path=os.path.join(DATA_DIR, "gaba_raw_counts.h5ad"),
        dataset_name="GABA (Lee & Dalley 2023)",
    )

    # -------------------------------------------------------------------
    # Excitatory dataset
    # -------------------------------------------------------------------
    exc_adata = build_h5ad(
        expression_csv_gz=os.path.join(DATA_DIR, "excitatory_expression.csv.gz"),
        metadata_csv=os.path.join(DATA_DIR, "excitatory_cell_metadata.csv"),
        gene_name_id=gene_name_id,
        token_ensembl_ids=token_ensembl_ids,
        output_path=os.path.join(DATA_DIR, "excitatory_raw_counts.h5ad"),
        dataset_name="Excitatory (Berg et al. 2021)",
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, adata in [("GABA", gaba_adata), ("Excitatory", exc_adata)]:
        print(f"\n{name}:")
        print(f"  Cells: {adata.n_obs}")
        print(f"  Genes (mapped to V1 token dict): {adata.n_vars}")
        print(f"  .X contains CPM values (converted from log2(CPM+1))")
        print(f"  n_counts: {adata.obs['n_counts'].iloc[0]:.0f} (all cells, CPM convention)")
        cpm_sums = adata.obs["cpm_sum_mapped_genes"]
        print(f"  CPM sum over mapped genes: min={cpm_sums.min():.0f}, "
              f"median={cpm_sums.median():.0f}, max={cpm_sums.max():.0f}")
        print(f"  Specimen IDs: {adata.obs['specimen_id'].nunique()} unique")
        # Coverage: how many of the 25,424 V1 genes are covered?
        n_v1_covered = len(set(adata.var["ensembl_id"]) & token_ensembl_ids)
        print(f"  V1 token dict coverage: {n_v1_covered}/{len(token_ensembl_ids)} "
              f"({100*n_v1_covered/len(token_ensembl_ids):.1f}%)")

    print("\nDone. h5ad files are ready for Geneformer tokenization.")


if __name__ == "__main__":
    main()
