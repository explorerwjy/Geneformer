# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Task 4: Tokenize Chaffin and Fetal Cardiomyocyte Datasets
#
# This notebook preprocesses and tokenizes two datasets for the Geneformer
# cardiomyopathy reproduction:
#
# 1. **Chaffin et al.** adult cardiomyocytes (NF/HCM/DCM) - 158,469 cells
# 2. **Cao et al.** fetal cardiomyocytes - 67,610 cells
#
# Preprocessing steps:
# - Filter to cardiomyocytes only (Chaffin)
# - Ensure Ensembl IDs in .var index
# - Add `n_counts` column
# - Save cleaned h5ad to separate directories (one file per directory)
# - Tokenize with Geneformer V2 settings (4096 input, special tokens)

# %%
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

# %%
# Project paths
PROJECT_ROOT = Path("/home/jw3514/Work/Geneformer/Geneformer")
DATA_DIR = PROJECT_ROOT / "data"

# Input paths
CHAFFIN_H5AD = (
    DATA_DIR
    / "chaffin_cardiomyopathy"
    / "SCP1303"
    / "anndata"
    / "human_dcm_hcm_scportal_03.17.2022.h5ad"
)
FETAL_H5AD = DATA_DIR / "fetal_cardiomyocytes" / "cao2020_fetal_cardiomyocytes.h5ad"

# Output paths for cleaned h5ad files (one file per directory for tokenizer)
CHAFFIN_CLEAN_DIR = DATA_DIR / "chaffin_cardiomyopathy" / "clean"
FETAL_CLEAN_DIR = DATA_DIR / "fetal_cardiomyocytes" / "clean"
TOKENIZED_DIR = DATA_DIR / "tokenized"

for d in [CHAFFIN_CLEAN_DIR, FETAL_CLEAN_DIR, TOKENIZED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Preprocess Chaffin Adult Cardiomyocytes

# %%
# Load the full Chaffin dataset (13 GB, ~593K cells)
# Using backed mode to inspect, then loading only cardiomyocytes
print("Loading Chaffin h5ad in backed mode to get cardiomyocyte indices...")
adata_backed = ad.read_h5ad(str(CHAFFIN_H5AD), backed="r")
print(f"Full dataset shape: {adata_backed.shape}")

# Identify cardiomyocyte indices
cm_types = ["Cardiomyocyte_I", "Cardiomyocyte_II", "Cardiomyocyte_III"]
cm_mask = adata_backed.obs["cell_type_leiden0.6"].isin(cm_types)
cm_indices = np.where(cm_mask.values)[0]
print(f"Cardiomyocyte cells: {len(cm_indices)}")
for ct in cm_types:
    n = (adata_backed.obs["cell_type_leiden0.6"] == ct).sum()
    print(f"  {ct}: {n}")

# %%
# Load only cardiomyocyte rows (much less memory than full 13GB)
# We read from backed mode by subsetting
print("Extracting cardiomyocyte subset from backed object...")
adata_cm = adata_backed[cm_indices].to_memory()
print(f"Cardiomyocyte subset shape: {adata_cm.shape}")

# Close backed file
del adata_backed

# %%
# Verify raw integer counts
if sp.issparse(adata_cm.X):
    sample = adata_cm.X[:10].toarray()
else:
    sample = adata_cm.X[:10]
nz = sample[sample != 0][:20]
print(f"Sample nonzero values: {nz}")
print(f"Are integers: {np.allclose(nz, np.round(nz))}")

# %%
# Set up gene IDs: var index is gene symbols, gene_ids column has Ensembl IDs
# The tokenizer needs ensembl_id in .var (either as column or index)
print(f"var index[:5]: {list(adata_cm.var.index[:5])}")
print(f"gene_ids[:5]: {list(adata_cm.var['gene_ids'][:5])}")

# Set the Ensembl IDs as the var index so we can use use_h5ad_index=True
adata_cm.var["gene_symbol"] = adata_cm.var.index.copy()
adata_cm.var.index = adata_cm.var["gene_ids"].values
adata_cm.var.index.name = None
print(f"New var index[:5]: {list(adata_cm.var.index[:5])}")

# %%
# Add n_counts column
# cellbender_ncount matches the sum of X (verified earlier)
adata_cm.obs["n_counts"] = adata_cm.obs["cellbender_ncount"].values.astype(np.float32)
print(f"n_counts range: {adata_cm.obs['n_counts'].min():.0f} - {adata_cm.obs['n_counts'].max():.0f}")

# Verify
if sp.issparse(adata_cm.X):
    row_sums = np.array(adata_cm.X.sum(axis=1)).flatten()
else:
    row_sums = adata_cm.X.sum(axis=1)
diff = np.abs(row_sums - adata_cm.obs["n_counts"].values)
print(f"Max difference between X.sum and n_counts: {diff.max():.2f}")

# %%
# Check disease and cell type labels
print("Disease labels:", adata_cm.obs["disease"].unique().tolist())
print("Cell type labels:", adata_cm.obs["cell_type_leiden0.6"].unique().tolist())

# Labels are already in good format:
# disease: NF, HCM, DCM
# cell_type: Cardiomyocyte_I, Cardiomyocyte_II, Cardiomyocyte_III

# %%
# Save cleaned Chaffin cardiomyocyte h5ad
chaffin_clean_path = CHAFFIN_CLEAN_DIR / "chaffin_cardiomyocytes.h5ad"
print(f"Saving cleaned Chaffin h5ad to {chaffin_clean_path}...")
adata_cm.write_h5ad(str(chaffin_clean_path))
print(f"Saved. File size: {chaffin_clean_path.stat().st_size / 1e9:.2f} GB")

# Free memory
del adata_cm

# %% [markdown]
# ## 2. Preprocess Fetal Cardiomyocytes (Cao et al.)

# %%
print("Loading fetal cardiomyocyte h5ad...")
adata_fetal = ad.read_h5ad(str(FETAL_H5AD))
print(f"Shape: {adata_fetal.shape}")
print(f"var index[:5]: {list(adata_fetal.var.index[:5])}")
print(f"feature_id[:5]: {list(adata_fetal.var['feature_id'][:5])}")
print(f"cell_type unique: {adata_fetal.obs['cell_type'].unique().tolist()}")

# %%
# Set Ensembl IDs as var index
adata_fetal.var["original_index"] = adata_fetal.var.index.copy()
adata_fetal.var.index = adata_fetal.var["feature_id"].values
adata_fetal.var.index.name = None
print(f"New var index[:5]: {list(adata_fetal.var.index[:5])}")

# %%
# Add n_counts from raw_sum
adata_fetal.obs["n_counts"] = adata_fetal.obs["raw_sum"].values.astype(np.float32)
print(f"n_counts range: {adata_fetal.obs['n_counts'].min():.0f} - {adata_fetal.obs['n_counts'].max():.0f}")

# Verify
if sp.issparse(adata_fetal.X):
    row_sums = np.array(adata_fetal.X.sum(axis=1)).flatten()
else:
    row_sums = adata_fetal.X.sum(axis=1)
diff = np.abs(row_sums - adata_fetal.obs["n_counts"].values)
print(f"Max difference between X.sum and n_counts: {diff.max():.2f}")

# %%
# Verify raw counts
if sp.issparse(adata_fetal.X):
    sample = adata_fetal.X[:10].toarray()
else:
    sample = adata_fetal.X[:10]
nz = sample[sample != 0][:20]
print(f"Sample nonzero values: {nz}")
print(f"Are integers: {np.allclose(nz, np.round(nz))}")

# %%
# Save cleaned fetal h5ad
fetal_clean_path = FETAL_CLEAN_DIR / "fetal_cardiomyocytes.h5ad"
print(f"Saving cleaned fetal h5ad to {fetal_clean_path}...")
adata_fetal.write_h5ad(str(fetal_clean_path))
print(f"Saved. File size: {fetal_clean_path.stat().st_size / 1e6:.1f} MB")

del adata_fetal

# %% [markdown]
# ## 3. Tokenize Both Datasets with Geneformer V2

# %%
from geneformer import TranscriptomeTokenizer

# %% [markdown]
# ### 3a. Tokenize Chaffin Adult Cardiomyocytes
#
# Custom attributes to preserve:
# - `disease`: NF/HCM/DCM
# - `cell_type_leiden0.6`: Cardiomyocyte_I/II/III
# - `donor_id`: patient IDs

# %%
print("Tokenizing Chaffin cardiomyocytes...")
tk_chaffin = TranscriptomeTokenizer(
    custom_attr_name_dict={
        "disease": "disease",
        "cell_type_leiden0.6": "cell_type",
        "donor_id": "donor_id",
    },
    nproc=10,
    model_input_size=4096,
    special_token=True,
    model_version="V2",
    use_h5ad_index=True,
)
tk_chaffin.tokenize_data(
    data_directory=str(CHAFFIN_CLEAN_DIR),
    output_directory=str(TOKENIZED_DIR),
    output_prefix="chaffin_cardiomyocytes",
    file_format="h5ad",
)
print("Chaffin tokenization complete.")

# %% [markdown]
# ### 3b. Tokenize Fetal Cardiomyocytes

# %%
print("Tokenizing fetal cardiomyocytes...")
tk_fetal = TranscriptomeTokenizer(
    custom_attr_name_dict={
        "cell_type": "cell_type",
        "donor_id": "donor_id",
        "development_stage": "development_stage",
    },
    nproc=10,
    model_input_size=4096,
    special_token=True,
    model_version="V2",
    use_h5ad_index=True,
)
tk_fetal.tokenize_data(
    data_directory=str(FETAL_CLEAN_DIR),
    output_directory=str(TOKENIZED_DIR),
    output_prefix="fetal_cardiomyocytes",
    file_format="h5ad",
)
print("Fetal tokenization complete.")

# %% [markdown]
# ## 4. Validate Tokenized Datasets

# %%
from datasets import load_from_disk

print("=" * 60)
print("VALIDATION")
print("=" * 60)

# Chaffin
chaffin_ds = load_from_disk(str(TOKENIZED_DIR / "chaffin_cardiomyocytes.dataset"))
print(f"\nChaffin dataset:")
print(f"  Columns: {chaffin_ds.column_names}")
print(f"  Cells: {len(chaffin_ds)}")
print(f"  Sample input_ids length: {chaffin_ds[0]['length']}")
print(f"  Disease values: {sorted(set(chaffin_ds['disease']))}")
print(f"  Cell types: {sorted(set(chaffin_ds['cell_type']))}")
print(f"  Donors: {len(set(chaffin_ds['donor_id']))}")

# Fetal
fetal_ds = load_from_disk(str(TOKENIZED_DIR / "fetal_cardiomyocytes.dataset"))
print(f"\nFetal dataset:")
print(f"  Columns: {fetal_ds.column_names}")
print(f"  Cells: {len(fetal_ds)}")
print(f"  Sample input_ids length: {fetal_ds[0]['length']}")
print(f"  Cell types: {sorted(set(fetal_ds['cell_type']))}")
print(f"  Dev stages: {sorted(set(fetal_ds['development_stage']))}")
print(f"  Donors: {len(set(fetal_ds['donor_id']))}")

# Quick sanity checks
assert len(chaffin_ds) > 150000, f"Expected >150K chaffin cells, got {len(chaffin_ds)}"
assert len(fetal_ds) > 60000, f"Expected >60K fetal cells, got {len(fetal_ds)}"
assert set(chaffin_ds["disease"]) == {"NF", "HCM", "DCM"}, "Missing disease labels"
assert "input_ids" in chaffin_ds.column_names
assert "input_ids" in fetal_ds.column_names

print("\nAll validation checks passed!")
