# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Download Fetal Cardiomyocyte Data (Cao et al. 2020)
#
# Source: "A human cell atlas of fetal gene expression" (Science 370, 2020)
# GEO accession: GSE156793
# DOI: 10.1126/science.aba7721
#
# We retrieve fetal cardiac muscle cells from the CELLxGENE Census API,
# which hosts a curated version of the full Cao et al. dataset (~4M cells).
# We filter for heart tissue, cardiac muscle cell type only.
#
# These cells will be used for zero-shot in silico deletion analysis with
# Geneformer V2-316M to identify network regulators in cardiomyocyte identity.

# %%
import os
import traceback

import anndata as ad
import cellxgene_census
import numpy as np

# %%
# Configuration
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "fetal_cardiomyocytes",
)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cao2020_fetal_cardiomyocytes.h5ad")

# Cao et al. 2020 dataset IDs in CELLxGENE Census
# Full dataset (~4M cells): f7c1c579-2dc0-47e2-ba19-8165c5a0e353
# 1M subset: fa27492b-82ff-4ab7-ac61-0e2b184eee67
CAO_FULL_DATASET_ID = "f7c1c579-2dc0-47e2-ba19-8165c5a0e353"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ## Download from CELLxGENE Census
#
# We query for:
# - dataset_id = Cao et al. full dataset
# - tissue_general = "heart"
# - cell_type = "cardiac muscle cell"
#
# The Census uses the ontology term "cardiac muscle cell" for cardiomyocytes.

# %%
if os.path.exists(OUTPUT_PATH):
    print(f"File already exists: {OUTPUT_PATH}")
    print("Loading existing file...")
    adata = ad.read_h5ad(OUTPUT_PATH)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
else:
    print("Opening CELLxGENE Census...")
    with cellxgene_census.open_soma() as census:
        print("Downloading cardiac muscle cells from Cao et al. 2020...")
        adata = cellxgene_census.get_anndata(
            census,
            "Homo sapiens",
            obs_value_filter=(
                f"dataset_id == '{CAO_FULL_DATASET_ID}' "
                "and tissue_general == 'heart' "
                "and cell_type == 'cardiac muscle cell'"
            ),
        )
        print(f"Downloaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

        # Save
        print(f"Saving to {OUTPUT_PATH}...")
        adata.write_h5ad(OUTPUT_PATH)
        fsize = os.path.getsize(OUTPUT_PATH) / 1e6
        print(f"Saved ({fsize:.1f} MB)")

# %% [markdown]
# ## Inspect the data

# %%
print("=" * 60)
print("FETAL CARDIOMYOCYTE DATA SUMMARY")
print("=" * 60)
print(f"\nShape: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"Cell type: {adata.obs['cell_type'].unique().tolist()}")
print(f"Tissue: {adata.obs['tissue'].unique().tolist()}")
print(f"Disease status: {adata.obs['disease'].unique().tolist()}")
print(f"Assay: {adata.obs['assay'].unique().tolist()}")
print(f"Suspension type: {adata.obs['suspension_type'].unique().tolist()}")

# %%
# Development stages (gestational age)
print("\n--- Development stages ---")
dev_counts = adata.obs["development_stage"].value_counts()
dev_counts = dev_counts[dev_counts > 0]
for stage, count in dev_counts.items():
    print(f"  {stage}: {count:,} cells")

# %%
# Donor breakdown
print(f"\n--- Donors ({adata.obs['donor_id'].nunique()} total) ---")
donor_counts = adata.obs["donor_id"].value_counts()
for donor, count in donor_counts.items():
    if count > 0:
        print(f"  {donor}: {count:,} cells")

# %%
# Sex distribution
print("\n--- Sex distribution ---")
sex_counts = adata.obs["sex"].value_counts()
for sex, count in sex_counts.items():
    if count > 0:
        print(f"  {sex}: {count:,} cells")

# %%
# Gene annotation format
print("\n--- Gene annotations ---")
print(f"  Total genes: {len(adata.var)}")
print(f"  var columns: {list(adata.var.columns)}")
print(f"  feature_id examples (Ensembl): {list(adata.var['feature_id'][:5])}")
print(f"  feature_name examples (symbol): {list(adata.var['feature_name'][:5])}")
ensembl_count = adata.var["feature_id"].str.startswith("ENSG").sum()
print(f"  Genes with ENSG IDs: {ensembl_count} / {len(adata.var)} (all)")

# %%
# Expression statistics
print("\n--- Expression statistics ---")
total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
genes_detected = np.asarray((adata.X > 0).sum(axis=1)).flatten()
print(f"  Total counts per cell:")
print(f"    mean={total_counts.mean():.1f}, median={np.median(total_counts):.1f}")
print(f"    min={total_counts.min():.1f}, max={total_counts.max():.1f}")
print(f"  Genes detected per cell:")
print(f"    mean={genes_detected.mean():.1f}, median={np.median(genes_detected):.1f}")
print(f"    min={genes_detected.min():.0f}, max={genes_detected.max():.0f}")

# %% [markdown]
# ## Notes for downstream processing
#
# Key observations:
# - **67,610 cardiac muscle cells** from fetal heart tissue (Cao et al. 2020)
# - Gestational ages: 12-17 weeks post-fertilization (5 stages)
# - 9 donors, mix of male and female
# - Assay: sci-RNA-seq3 (single-nucleus)
# - All cells annotated as "normal" (no disease)
# - Gene IDs are Ensembl (ENSG*) format -- compatible with Geneformer tokenizer
# - Relatively low total counts per cell (median ~384), typical for sci-RNA-seq3
# - The tokenizer uses rank value encoding, so absolute count depth matters less
#
# For tokenization:
# - The `feature_id` column contains Ensembl gene IDs needed by Geneformer
# - May need to set var_names to `feature_id` before tokenizing
# - Geneformer V2-316M expects up to 4096 input tokens (genes per cell)
# - With median ~286 genes detected per cell, all genes fit within the window
