"""
Convert exported Seurat counts to h5ad format for Geneformer tokenization.
Maps gene symbols to Ensembl IDs using Geneformer's gene dictionaries.
"""

import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io

# ── Load exported data ──────────────────────────────────────────────
print("Loading counts matrix...")
counts = scipy.io.mmread(
    "/home/jw3514/Work/Geneformer/data/QiuyanLiver/counts.mtx"
).tocsr()
print(f"  Shape: {counts.shape} (genes x cells)")

genes = pd.read_csv("/home/jw3514/Work/Geneformer/data/QiuyanLiver/genes.csv")[
    "gene"
].values
barcodes = pd.read_csv("/home/jw3514/Work/Geneformer/data/QiuyanLiver/barcodes.csv")[
    "barcode"
].values
meta = pd.read_csv(
    "/home/jw3514/Work/Geneformer/data/QiuyanLiver/metadata.csv", index_col=0
)

print(f"  Genes: {len(genes)}, Cells: {len(barcodes)}")

# ── Load Geneformer gene dictionaries ───────────────────────────────
print("\nLoading Geneformer gene dictionaries...")
gf_dir = "/home/jw3514/Work/Geneformer/Geneformer/geneformer"

with open(f"{gf_dir}/gene_name_id_dict_gc104M.pkl", "rb") as f:
    gene_name_to_id = pickle.load(f)  # symbol -> ensembl

with open(f"{gf_dir}/ensembl_mapping_dict_gc104M.pkl", "rb") as f:
    ensembl_mapping = pickle.load(f)  # broader mapping (case-insensitive names)

with open(f"{gf_dir}/token_dictionary_gc104M.pkl", "rb") as f:
    token_dict = pickle.load(f)  # ensembl -> token_id

print(f"  gene_name_id_dict: {len(gene_name_to_id)} entries")
print(f"  ensembl_mapping_dict: {len(ensembl_mapping)} entries")
print(f"  token_dict: {len(token_dict)} Geneformer tokens")

# ── Map gene symbols to Ensembl IDs ────────────────────────────────
print("\nMapping gene symbols to Ensembl IDs...")

# Build combined lookup: try gene_name_id_dict first, then ensembl_mapping
# The ensembl_mapping_dict uses UPPERCASE keys
ensembl_ids = []
mapped_count = 0
for g in genes:
    eid = None
    # Try exact match in gene_name_id_dict
    if g in gene_name_to_id:
        eid = gene_name_to_id[g]
    # Try uppercase in ensembl_mapping
    elif g.upper() in ensembl_mapping:
        eid = ensembl_mapping[g.upper()]
    # Try without version suffix (e.g., AL627309.1 -> AL627309)
    elif "." in g and g.rsplit(".", 1)[0].upper() in ensembl_mapping:
        eid = ensembl_mapping[g.rsplit(".", 1)[0].upper()]

    if eid is not None and eid in token_dict:
        ensembl_ids.append(eid)
        mapped_count += 1
    else:
        ensembl_ids.append(None)

print(f"  Mapped to Geneformer tokens: {mapped_count}/{len(genes)} genes")
print(f"  Unmapped: {len(genes) - mapped_count}")

# Show some unmapped examples
unmapped = [(g, ensembl_ids[i]) for i, g in enumerate(genes) if ensembl_ids[i] is None]
print(f"  Unmapped examples: {[u[0] for u in unmapped[:10]]}")

# ── Filter to mapped genes only ─────────────────────────────────────
mask = np.array([eid is not None for eid in ensembl_ids])
filtered_genes = genes[mask]
filtered_ensembl = [eid for eid in ensembl_ids if eid is not None]
filtered_counts = counts[mask, :]  # genes x cells, keep only mapped genes

print(f"\n  After filtering: {filtered_counts.shape[0]} genes, {filtered_counts.shape[1]} cells")

# Check for duplicate Ensembl IDs (Geneformer's collapse_gene_ids handles this)
unique_ensembl = set(filtered_ensembl)
print(f"  Unique Ensembl IDs: {len(unique_ensembl)}")
if len(unique_ensembl) < len(filtered_ensembl):
    print(f"  ({len(filtered_ensembl) - len(unique_ensembl)} duplicates - Geneformer will collapse)")

# ── Build AnnData (cells x genes) ──────────────────────────────────
print("\nBuilding AnnData object...")
# Transpose: cells x genes
X = filtered_counts.T.tocsr()
print(f"  X shape: {X.shape}")

# obs (cell metadata)
obs = meta.copy()
obs["n_counts"] = np.array(X.sum(axis=1)).flatten()
print(f"  n_counts range: {obs['n_counts'].min():.0f} - {obs['n_counts'].max():.0f}")

# var (gene metadata)
var = pd.DataFrame(
    {"gene_symbol": filtered_genes, "ensembl_id": filtered_ensembl},
    index=filtered_ensembl,
)

# Create AnnData
adata = ad.AnnData(X=X, obs=obs, var=var)
adata.obs_names = barcodes
adata.var_names = filtered_ensembl

print(f"\n  AnnData: {adata.shape}")
print(f"  obs columns: {list(adata.obs.columns)}")
print(f"  var columns: {list(adata.var.columns)}")
print(f"  Cell types: {adata.obs['Cell.type.new'].nunique()}")
print(adata.obs["Cell.type.new"].value_counts())
print(f"\n  Samples: {adata.obs['orig.ident'].nunique()}")
print(adata.obs["orig.ident"].value_counts())

# ── Save ────────────────────────────────────────────────────────────
out_path = "/home/jw3514/Work/Geneformer/data/QiuyanLiver/liver_raw_counts.h5ad"
print(f"\nSaving to {out_path}...")
adata.write_h5ad(out_path)
print("Done!")
