# Cardiomyopathy Reproduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce Geneformer cardiomyopathy experiments (zero-shot deletion, disease classification, in silico treatment, per-gene impact) using V2-316M, with adult (Chaffin et al.) and fetal (Cao et al.) cardiomyocyte datasets.

**Architecture:** Download raw scRNA-seq from Broad SCP1303 (adult) and GEO GSE156793 (fetal), tokenize with V2 settings (4096 input, special tokens, gc104M dictionaries), then run four analyses: zero-shot in silico deletion on fetal cardiomyocytes, fine-tune classifier on adult NF/HCM/DCM, in silico treatment analysis to find therapeutic TFs, and per-gene impact extraction for ion channel genes.

**Tech Stack:** Geneformer (this repo), PyTorch + transformers, HuggingFace datasets, scanpy/anndata, matplotlib/seaborn for plotting. Conda env: `geneformer`. GPU required.

---

## Task 1: Set Up Directory Structure and Verify Environment

**Files:**
- Create: `scripts/cm_reproduction/00_setup.py`

**Step 1: Create output directory structure**

```bash
cd /home/jw3514/Work/Geneformer/Geneformer
mkdir -p data/chaffin_cardiomyopathy
mkdir -p data/fetal_cardiomyocytes
mkdir -p data/tokenized
mkdir -p data/gene_lists
mkdir -p outputs/phase1_tokenization
mkdir -p outputs/phase2a_zero_shot_deletion
mkdir -p outputs/phase2b_classification
mkdir -p outputs/phase2c_treatment_analysis
mkdir -p outputs/phase2d_per_gene_impact
mkdir -p scripts/cm_reproduction
```

**Step 2: Verify environment**

```bash
conda activate geneformer
python -c "
import geneformer
import torch
import scanpy
import anndata
print(f'Geneformer imported from: {geneformer.__file__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'Scanpy version: {scanpy.__version__}')
"
```

Expected: All imports succeed, CUDA available, at least 1 GPU.

**Step 3: Verify V2-316M model exists**

```bash
ls -la /home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-316M/
```

Expected: Model directory with `config.json`, `model.safetensors` or `pytorch_model.bin`.

**Step 4: Verify V2 token dictionaries**

```bash
conda activate geneformer
python -c "
from geneformer import GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE, ENSEMBL_DICTIONARY_FILE, ENSEMBL_MAPPING_FILE
import pickle
for name, path in [('GENE_MEDIAN', GENE_MEDIAN_FILE), ('TOKEN_DICT', TOKEN_DICTIONARY_FILE),
                    ('ENSEMBL_DICT', ENSEMBL_DICTIONARY_FILE), ('ENSEMBL_MAP', ENSEMBL_MAPPING_FILE)]:
    with open(path, 'rb') as f:
        d = pickle.load(f)
    print(f'{name}: {len(d)} entries, path={path}')
"
```

Expected: ~20K entries for TOKEN_DICT, ~20K for GENE_MEDIAN.

**Step 5: Commit setup**

```bash
git add scripts/cm_reproduction/
git commit -m "Add directory structure for cardiomyopathy reproduction"
```

---

## Task 2: Download and Inspect Adult Cardiomyocyte Data (Chaffin et al.)

**Files:**
- Create: `scripts/cm_reproduction/01a_download_chaffin.py`

**Step 1: Check if pre-tokenized V1 data exists locally**

The HuggingFace repo has a pre-tokenized V1 dataset at `data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset`. However, this was tokenized with V1 settings (2048 input, gc30M dictionaries) and **cannot be used directly with V2-316M** because the token IDs differ. We need the raw data for re-tokenization.

Check if the pre-tokenized V1 data is already available locally:

```bash
ls -la /home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/ 2>/dev/null || echo "V1 pre-tokenized data not found locally"
```

**Step 2: Download raw Chaffin et al. data from Broad Single Cell Portal**

The data is at Broad SCP1303. Download processed count matrix and metadata.

```bash
conda activate geneformer
# Option A: Check CZ CELLxGENE Census for h5ad (preferred, easiest format)
python -c "
import cellxgene_census
import tiledbsoma
# Search for Chaffin 2022 cardiomyopathy dataset
with cellxgene_census.open_soma() as census:
    datasets = census['census_info']['datasets'].read().concat().to_pandas()
    chaffin = datasets[datasets['dataset_title'].str.contains('cardiomyop', case=False, na=False) |
                       datasets['collection_name'].str.contains('Chaffin', case=False, na=False)]
    print(chaffin[['dataset_id', 'dataset_title', 'collection_name']].to_string())
" 2>/dev/null || echo "cellxgene_census not available, use alternative download"
```

If cellxgene_census is not available, download from Broad Single Cell Portal manually:

```bash
# Alternative: Download from Broad SCP1303
# Visit: https://singlecell.broadinstitute.org/single_cell/study/SCP1303/
# Download the expression matrix and metadata files
# Place in data/chaffin_cardiomyopathy/
```

If neither works, we can use the V1 pre-tokenized data as a **fallback** for Tasks 4-6 (classification and ISP), since the existing scripts already work with it. The key trade-off: V1-tokenized data uses different gene medians, so V2 model predictions may be suboptimal.

**Step 3: Write download/inspection script**

Create `scripts/cm_reproduction/01a_download_chaffin.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Download and inspect Chaffin et al. 2022 cardiomyopathy data.
Source: Broad Single Cell Portal SCP1303
Paper: Nature 608, 174-180 (2022)
"""
import os
import scanpy as sc
import pandas as pd
import numpy as np

DATA_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/chaffin_cardiomyopathy"
os.makedirs(DATA_DIR, exist_ok=True)

# %%
# --- Option 1: Load from cellxgene census ---
try:
    import cellxgene_census
    with cellxgene_census.open_soma() as census:
        # Query for heart tissue with cardiomyopathy
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=(
                "tissue_general == 'heart' and "
                "disease in ['dilated cardiomyopathy', 'hypertrophic cardiomyopathy', 'normal']"
            ),
            column_names={
                "obs": ["cell_type", "disease", "donor_id", "tissue", "dataset_id",
                        "assay", "suspension_type"],
            },
        )
    print(f"Downloaded {adata.n_obs} cells, {adata.n_vars} genes from CELLxGENE")
    print(f"Disease distribution:\n{adata.obs['disease'].value_counts()}")
    print(f"Cell types:\n{adata.obs['cell_type'].value_counts().head(20)}")

    # Save
    adata.write_h5ad(os.path.join(DATA_DIR, "chaffin_heart.h5ad"))
    print(f"Saved to {DATA_DIR}/chaffin_heart.h5ad")

except Exception as e:
    print(f"CELLxGENE download failed: {e}")
    print("Please download manually from Broad SCP1303")

# %%
# --- Option 2: Load from local h5ad if already downloaded ---
h5ad_path = os.path.join(DATA_DIR, "chaffin_heart.h5ad")
if os.path.exists(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"\nMetadata columns: {list(adata.obs.columns)}")
    print(f"\nDisease distribution:\n{adata.obs['disease'].value_counts()}")
    print(f"\nCell types:\n{adata.obs['cell_type'].value_counts().head(20)}")

    # Check for required columns
    required = ["disease", "cell_type"]
    for col in required:
        if col not in adata.obs.columns:
            print(f"WARNING: Missing column '{col}' - check alternative column names")
            print(f"Available columns: {list(adata.obs.columns)}")

    # Check for Ensembl IDs in gene names
    sample_genes = adata.var_names[:5].tolist()
    print(f"\nSample gene names: {sample_genes}")
    has_ensembl = any(g.startswith("ENSG") for g in sample_genes)
    print(f"Gene names are Ensembl IDs: {has_ensembl}")
    if not has_ensembl and "ensembl_id" in adata.var.columns:
        print("Ensembl IDs found in adata.var['ensembl_id']")
    elif not has_ensembl:
        print("WARNING: No Ensembl IDs found. Will need gene name -> Ensembl mapping.")
        print(f"Available var columns: {list(adata.var.columns)}")
```

**Step 4: Run the script and inspect output**

```bash
conda activate geneformer
python scripts/cm_reproduction/01a_download_chaffin.py
```

Expected: Either downloads data from CELLxGENE or prints instructions for manual download. We need to see the exact column names for disease, cell_type, and individual/donor_id to configure tokenization.

**Step 5: Commit**

```bash
git add scripts/cm_reproduction/01a_download_chaffin.py
git commit -m "Add Chaffin et al. data download and inspection script"
```

---

## Task 3: Download and Inspect Fetal Cardiomyocyte Data (Cao et al.)

**Files:**
- Create: `scripts/cm_reproduction/01b_download_fetal.py`

**Step 1: Write download script**

Create `scripts/cm_reproduction/01b_download_fetal.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Download and inspect Cao et al. 2020 fetal cell atlas data.
Source: GEO GSE156793
Paper: Science 370, eaba7721 (2020)
We need only the fetal cardiomyocyte subset.
"""
import os
import scanpy as sc
import pandas as pd
import numpy as np

DATA_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/fetal_cardiomyocytes"
os.makedirs(DATA_DIR, exist_ok=True)

# %%
# --- Option 1: Try CELLxGENE Census ---
try:
    import cellxgene_census
    with cellxgene_census.open_soma() as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=(
                "tissue_general == 'heart' and "
                "development_stage in ['human late first trimester stage', "
                "'human second trimester stage', 'human early first trimester stage', "
                "'fetal stage', '10th week post-fertilization human stage', "
                "'16th week post-fertilization human stage']"
            ),
            column_names={
                "obs": ["cell_type", "tissue", "donor_id", "dataset_id",
                        "development_stage", "assay"],
            },
        )
    print(f"Downloaded {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Cell types:\n{adata.obs['cell_type'].value_counts().head(20)}")
    print(f"Development stages:\n{adata.obs['development_stage'].value_counts()}")

    # Filter for cardiomyocytes
    cm_mask = adata.obs["cell_type"].str.contains("cardiomyocyte", case=False, na=False)
    adata_cm = adata[cm_mask].copy()
    print(f"\nFiltered to {adata_cm.n_obs} cardiomyocytes")

    adata_cm.write_h5ad(os.path.join(DATA_DIR, "fetal_cardiomyocytes.h5ad"))
    print(f"Saved to {DATA_DIR}/fetal_cardiomyocytes.h5ad")

except Exception as e:
    print(f"CELLxGENE download failed: {e}")
    print("\nFallback: Download from GEO GSE156793")
    print("wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156793/suppl/GSE156793_S3_gene_count.loom.gz")
    print(f"gunzip and place in {DATA_DIR}/")

# %%
# --- Inspect if already downloaded ---
for fname in ["fetal_cardiomyocytes.h5ad", "GSE156793_S3_gene_count.loom"]:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        print(f"\nInspecting {fname}...")
        if fname.endswith(".h5ad"):
            adata = sc.read_h5ad(fpath)
        else:
            adata = sc.read_loom(fpath)
        print(f"Shape: {adata.n_obs} cells x {adata.n_vars} genes")
        print(f"Metadata columns: {list(adata.obs.columns)}")
        print(f"Sample gene names: {adata.var_names[:5].tolist()}")
        break
```

**Step 2: Run and inspect**

```bash
conda activate geneformer
python scripts/cm_reproduction/01b_download_fetal.py
```

**Step 3: Commit**

```bash
git add scripts/cm_reproduction/01b_download_fetal.py
git commit -m "Add fetal cardiomyocyte data download script"
```

---

## Task 4: Tokenize Datasets with V2 Settings

**Files:**
- Create: `scripts/cm_reproduction/02_tokenize.py`

**Depends on:** Tasks 2-3 (data downloaded and inspected, column names known)

**Step 1: Write tokenization script**

Create `scripts/cm_reproduction/02_tokenize.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Tokenize cardiomyocyte datasets with Geneformer V2 settings.

IMPORTANT: Before running, update custom_attr_name_dict and file paths
based on actual column names found in Task 2/3 inspection.
"""
import os
from geneformer import TranscriptomeTokenizer

OUTPUT_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/tokenized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# --- Dataset A: Chaffin et al. adult cardiomyocytes ---
# NOTE: Update column names based on actual h5ad inspection from Task 2
# The keys are column names in the h5ad .obs, values are names in output dataset

CHAFFIN_DATA_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/chaffin_cardiomyopathy"
chaffin_custom_attrs = {
    "cell_type": "cell_type",       # UPDATE if column name differs
    "disease": "disease",           # UPDATE: might be "disease_state", "condition", etc.
    "donor_id": "individual",       # UPDATE: might be "individual", "patient_id", etc.
}

print("Tokenizing Chaffin et al. adult cardiomyocytes...")
tk_adult = TranscriptomeTokenizer(
    custom_attr_name_dict=chaffin_custom_attrs,
    nproc=10,
    model_input_size=4096,
    special_token=True,
    model_version="V2",
)
tk_adult.tokenize_data(
    data_directory=CHAFFIN_DATA_DIR,
    output_directory=OUTPUT_DIR,
    output_prefix="chaffin_cm",
    file_format="h5ad",
)
print("Chaffin tokenization complete!")

# %%
# --- Dataset B: Fetal cardiomyocytes ---
FETAL_DATA_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/fetal_cardiomyocytes"
fetal_custom_attrs = {
    "cell_type": "cell_type",       # UPDATE based on Task 3 inspection
}

print("Tokenizing fetal cardiomyocytes...")
tk_fetal = TranscriptomeTokenizer(
    custom_attr_name_dict=fetal_custom_attrs,
    nproc=10,
    model_input_size=4096,
    special_token=True,
    model_version="V2",
)
tk_fetal.tokenize_data(
    data_directory=FETAL_DATA_DIR,
    output_directory=OUTPUT_DIR,
    output_prefix="fetal_cm",
    file_format="h5ad",
)
print("Fetal tokenization complete!")

# %%
# --- Validate tokenized datasets ---
from datasets import load_from_disk

for name, prefix in [("Chaffin adult", "chaffin_cm"), ("Fetal", "fetal_cm")]:
    dataset_path = os.path.join(OUTPUT_DIR, f"{prefix}.dataset")
    if not os.path.exists(dataset_path):
        print(f"WARNING: {dataset_path} not found")
        continue

    ds = load_from_disk(dataset_path)
    print(f"\n{'='*60}")
    print(f"{name} dataset: {len(ds)} cells")
    print(f"Columns: {ds.column_names}")
    print(f"Sample input_ids length: {len(ds[0]['input_ids'])}")
    print(f"Sample attention_mask: {ds[0].get('attention_mask', 'N/A')}")

    # Check metadata preserved
    if "disease" in ds.column_names:
        from collections import Counter
        diseases = Counter(ds["disease"])
        print(f"Disease distribution: {dict(diseases)}")

    if "cell_type" in ds.column_names:
        from collections import Counter
        cell_types = Counter(ds["cell_type"])
        print(f"Cell type distribution: {dict(cell_types)}")
```

**Step 2: Run tokenization**

```bash
conda activate geneformer
python scripts/cm_reproduction/02_tokenize.py
```

Expected: Two `.dataset` directories created in `data/tokenized/`. Chaffin should have ~93K cardiomyocytes (or ~600K total cells if not pre-filtered). Fetal should have cardiomyocytes from fetal heart atlas.

**Step 3: Commit**

```bash
git add scripts/cm_reproduction/02_tokenize.py
git commit -m "Add V2 tokenization script for cardiomyocyte datasets"
```

---

## Task 5: Curate Gene Lists (Disease vs Control)

**Files:**
- Create: `scripts/cm_reproduction/03_gene_lists.py`

**Step 1: Write gene list curation script**

This script curates the gene sets needed for the zero-shot deletion analysis. The paper used:
- **Disease genes**: cardiomyopathy + structural heart disease genes (from Supplementary Tables 3-4, and Pirruccello et al. 2020 GWAS)
- **Control genes**: hyperlipidaemia genes (expressed in cardiomyocytes but affect other cell types)

Create `scripts/cm_reproduction/03_gene_lists.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Curate gene lists for in silico deletion analysis.

Sources:
- Cardiomyopathy genes: OMIM, literature, Geneformer paper Supp Tables
- Structural heart disease genes: Pirruccello et al. 2020 GWAS (Nat Commun 11, 2254)
- Control genes: hyperlipidaemia genes expressed in cardiomyocytes
- Ion channel genes: for openCARP mapping

All genes stored as Ensembl IDs for Geneformer compatibility.
"""
import os
import pickle
import pandas as pd
from geneformer import ENSEMBL_DICTIONARY_FILE

# Load gene name -> Ensembl ID mapping
with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
    gene_name_to_id = pickle.load(f)

# Reverse mapping
gene_id_to_name = {v: k for k, v in gene_name_to_id.items()}

OUTPUT_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/gene_lists"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# --- Known cardiomyopathy / structural heart disease genes ---
# Curated from OMIM, ClinVar, and cardiomyopathy literature
# These are genes whose mutations cause cardiomyopathy
cardiomyopathy_genes_names = [
    # Sarcomere genes (HCM/DCM)
    "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "MYL2", "MYL3", "ACTC1",
    # Titin (DCM)
    "TTN",
    # Z-disc genes
    "TCAP", "LDB3", "CSRP3", "ACTN2",
    # Desmosomal genes (ARVC)
    "DSP", "PKP2", "DSG2", "DSC2", "JUP",
    # Lamin (DCM)
    "LMNA",
    # Ion channel / calcium handling
    "SCN5A", "RYR2", "PLN", "CASQ2",
    # Nuclear envelope
    "EMD", "TMEM43",
    # Transcription factors in cardiac development
    "TBX5", "GATA4", "NKX2-5", "HAND2", "MEF2C", "FOXM1",
    # Other known cardiomyopathy genes
    "BAG3", "FLNC", "RBM20", "TNNC1", "DES", "VCL",
    "TAZ", "DTNA", "SGCD", "DMD", "LAMP2",
]

# --- Structural heart disease genes from GWAS ---
# From Pirruccello et al. 2020 (cardiac MRI GWAS)
structural_heart_genes_names = [
    "ALPK3", "BAG3", "CDKN1A", "FLNC", "MTSS1", "PLN",
    "SMARCB1", "STRN", "SWI5", "SYNPO2L", "TBX20", "TNNT2",
    "TTN", "VCL",
]

# --- Hyperlipidaemia control genes ---
# Genes involved in lipid metabolism, expressed in cardiomyocytes
# but whose pathology primarily affects other cell types
hyperlipidaemia_genes_names = [
    "LDLR", "PCSK9", "APOB", "APOE", "APOA1", "APOC3",
    "CETP", "LIPC", "LIPG", "LPL", "ABCA1", "ABCG1",
    "HMGCR", "NPC1L1", "SCARB1", "ANGPTL3", "ANGPTL4",
    "SORT1", "TRIB1", "GCKR",
]

# --- Ion channel genes for openCARP ---
ion_channel_genes_names = [
    "KCNQ1",    # IKs (Kv7.1)
    "SCN5A",    # INa (Nav1.5)
    "CACNA1C",  # ICaL (Cav1.2)
    "KCNH2",    # IKr (hERG)
    "KCNJ2",    # IK1 (Kir2.1)
    "RYR2",     # Ryanodine receptor
    "ATP2A2",   # SERCA2a
    "SLC8A1",   # NCX1
    "KCNA5",    # IKur (Kv1.5)
    "KCNJ11",   # KATP
    "CACNA1G",  # ICaT (Cav3.1)
    "HCN4",     # If (funny current)
]

# --- Known TF list (for focused perturbation in 2C) ---
# Human transcription factors - use a curated list
# This is a subset; full TF list can be loaded from AnimalTFDB or similar
cardiac_tf_names = [
    "GATA4", "GATA6", "TBX5", "TBX20", "NKX2-5", "HAND1", "HAND2",
    "MEF2A", "MEF2C", "MEF2D", "TEAD1", "TEAD4", "SRF", "MYOCD",
    "FOXM1", "FOXO1", "FOXO3", "IRX4", "PITX2", "ISL1",
    "MEIS1", "MEIS2", "HEY1", "HEY2", "NOTCH1",
    "KLF2", "KLF4", "KLF15", "ETS1", "ETS2", "ERG",
    "STAT3", "NFAT5", "NFATC1", "NFATC2", "NFATC4",
    "PPARA", "PPARGC1A", "ESRRA", "ESRRG", "NR2F2",
    "TP53", "RB1", "MYC", "JUN", "FOS", "EGR1",
    "SMAD2", "SMAD3", "SMAD4", "CTCF", "YAP1", "WWTR1",
]

# %%
# --- Convert gene names to Ensembl IDs ---
def names_to_ensembl(gene_names, gene_name_to_id):
    """Convert gene names to Ensembl IDs, report any missing."""
    ensembl_ids = []
    missing = []
    for name in gene_names:
        if name in gene_name_to_id:
            ensembl_ids.append(gene_name_to_id[name])
        else:
            missing.append(name)
    if missing:
        print(f"  WARNING: {len(missing)} genes not found in V2 dictionary: {missing}")
    return ensembl_ids

print("Converting gene lists to Ensembl IDs...")

gene_lists = {
    "cardiomyopathy_genes": cardiomyopathy_genes_names,
    "structural_heart_genes": structural_heart_genes_names,
    "hyperlipidaemia_control": hyperlipidaemia_genes_names,
    "ion_channel_genes": ion_channel_genes_names,
    "cardiac_tfs": cardiac_tf_names,
}

ensembl_gene_lists = {}
for list_name, names in gene_lists.items():
    print(f"\n{list_name}:")
    ids = names_to_ensembl(names, gene_name_to_id)
    ensembl_gene_lists[list_name] = ids
    print(f"  {len(ids)}/{len(names)} genes mapped to Ensembl IDs")

# Combine disease genes (cardiomyopathy + structural)
disease_ensembl = list(set(
    ensembl_gene_lists["cardiomyopathy_genes"] +
    ensembl_gene_lists["structural_heart_genes"]
))
ensembl_gene_lists["disease_combined"] = disease_ensembl
print(f"\nCombined disease genes: {len(disease_ensembl)} unique Ensembl IDs")

# %%
# --- Save gene lists ---
for list_name, ids in ensembl_gene_lists.items():
    outpath = os.path.join(OUTPUT_DIR, f"{list_name}.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(ids, f)
    print(f"Saved {list_name}: {len(ids)} genes -> {outpath}")

# Also save as CSV for human readability
for list_name, names in gene_lists.items():
    ids = ensembl_gene_lists[list_name]
    df = pd.DataFrame({
        "gene_name": names[:len(ids)],
        "ensembl_id": ids,
    })
    df.to_csv(os.path.join(OUTPUT_DIR, f"{list_name}.csv"), index=False)

print("\nAll gene lists saved!")
```

**Step 2: Run and validate**

```bash
conda activate geneformer
python scripts/cm_reproduction/03_gene_lists.py
```

Expected: Gene list pickles and CSVs in `data/gene_lists/`. Most genes should map successfully; a few may be missing from the V2 dictionary (report which ones).

**Step 3: Commit**

```bash
git add scripts/cm_reproduction/03_gene_lists.py
git commit -m "Add gene list curation for cardiomyopathy analysis"
```

---

## Task 6: Zero-Shot In Silico Deletion on Fetal Cardiomyocytes (Phase 2A)

**Files:**
- Create: `scripts/cm_reproduction/04_zero_shot_deletion.py`

**Depends on:** Tasks 4 (tokenized fetal data), 5 (gene lists)

**Step 1: Write the zero-shot deletion script**

Create `scripts/cm_reproduction/04_zero_shot_deletion.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Phase 2A: Zero-shot in silico deletion on fetal cardiomyocytes.
Replicates Fig. 2d from Theodoris et al. 2023.

Uses pretrained V2-316M (no fine-tuning) to delete known cardiomyopathy
vs control (hyperlipidaemia) genes and compare embedding shifts.
"""
import matplotlib
matplotlib.use("Agg")
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from geneformer import InSilicoPerturber, InSilicoPerturberStats

# --- Paths ---
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-316M"
FETAL_DATA = "/home/jw3514/Work/Geneformer/Geneformer/data/tokenized/fetal_cm.dataset"
GENE_LIST_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/gene_lists"
OUTPUT_DIR = "/home/jw3514/Work/Geneformer/Geneformer/outputs/phase2a_zero_shot_deletion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load gene lists
with open(os.path.join(GENE_LIST_DIR, "disease_combined.pkl"), "rb") as f:
    disease_genes = pickle.load(f)
with open(os.path.join(GENE_LIST_DIR, "hyperlipidaemia_control.pkl"), "rb") as f:
    control_genes = pickle.load(f)

all_genes = list(set(disease_genes + control_genes))
print(f"Disease genes: {len(disease_genes)}")
print(f"Control genes: {len(control_genes)}")
print(f"Total genes to perturb: {len(all_genes)}")

# %%
# --- Step 1: Run in silico deletion (disease genes) ---
# NOTE: filter_data must match actual cell type labels in fetal dataset.
# UPDATE the cell type filter based on Task 3 inspection.
filter_data_dict = {"cell_type": ["cardiomyocyte"]}  # UPDATE as needed

print("Running in silico deletion for disease genes...")
isp_disease = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=disease_genes,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data=filter_data_dict,
    max_ncells=2000,
    emb_layer=-1,  # 2nd to last layer (more general representation)
    forward_batch_size=200,
    model_version="V2",
    nproc=10,
)

isp_disease.perturb_data(
    MODEL_DIR,
    FETAL_DATA,
    OUTPUT_DIR,
    "disease_genes",
)
print("Disease gene deletion complete!")

# %%
# --- Step 2: Run in silico deletion (control genes) ---
print("Running in silico deletion for control genes...")
isp_control = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=control_genes,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data=filter_data_dict,
    max_ncells=2000,
    emb_layer=-1,
    forward_batch_size=200,
    model_version="V2",
    nproc=10,
)

isp_control.perturb_data(
    MODEL_DIR,
    FETAL_DATA,
    OUTPUT_DIR,
    "control_genes",
)
print("Control gene deletion complete!")

# %%
# --- Step 3: Compute statistics ---
print("Computing statistics...")

# Use vs_null mode: compare disease gene effects vs control gene effects
ispstats = InSilicoPerturberStats(
    mode="vs_null",
    genes_perturbed=disease_genes,
    combos=0,
    model_version="V2",
)

ispstats.get_stats(
    input_data_directory=OUTPUT_DIR,
    null_dist_data_directory=OUTPUT_DIR,  # control genes serve as null
    output_directory=OUTPUT_DIR,
    output_prefix="disease_vs_control",
)
print("Statistics complete!")

# %%
# --- Step 4: Plot results (replicate Fig. 2d) ---
stats_file = os.path.join(OUTPUT_DIR, "disease_vs_control.csv")
if os.path.exists(stats_file):
    df = pd.read_csv(stats_file)
    print(f"\nResults shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTop 10 most deleterious genes:")
    print(df.nsmallest(10, "Test_vs_null_avg_shift")[["Gene_name", "Test_vs_null_avg_shift", "Test_vs_null_FDR"]])

    # Box plot: disease vs control cosine similarity
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # This is a simplified version - actual Fig 2d uses raw cosine similarities
    ax.set_title("In silico deletion effect on cardiomyocyte embeddings")
    ax.set_ylabel("Cosine similarity shift")
    # Plot will be populated once we have actual data
    fig.savefig(os.path.join(OUTPUT_DIR, "deletion_effect_boxplot.png"),
                dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Plot saved to {OUTPUT_DIR}/deletion_effect_boxplot.png")
else:
    print(f"Stats file not found at {stats_file}")

print("\nPhase 2A complete!")
```

**Step 2: Run the zero-shot deletion**

```bash
conda activate geneformer
python scripts/cm_reproduction/04_zero_shot_deletion.py
```

Expected: This is computationally intensive (~1-4 hours). Output: pickle files with cosine similarities and a CSV with statistics comparing disease vs control gene effects.

**Step 3: Commit**

```bash
git add scripts/cm_reproduction/04_zero_shot_deletion.py
git commit -m "Add zero-shot in silico deletion script (Phase 2A)"
```

---

## Task 7: Fine-Tune Disease Classifier (Phase 2B)

**Files:**
- Create: `scripts/cm_reproduction/05_disease_classification.py`

**Depends on:** Task 4 (tokenized Chaffin data)

**Step 1: Write classification script**

Create `scripts/cm_reproduction/05_disease_classification.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Phase 2B: Fine-tune V2-316M to classify NF vs HCM vs DCM cardiomyocytes.
Replicates Fig. 6a-b from Theodoris et al. 2023.

Uses Chaffin et al. 2022 adult cardiomyocyte data.
Patient-level split to avoid data leakage.
"""
import datetime
import glob
import os
from geneformer import Classifier

# --- Paths ---
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-316M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/Geneformer/data/tokenized/chaffin_cm.dataset"
# NOTE: If V2-tokenized data is not available, fall back to V1 pre-tokenized:
# INPUT_DATA = "/home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"

OUTPUT_DIR = f"/home/jw3514/Work/Geneformer/Geneformer/outputs/phase2b_classification/{datestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_prefix = "cm_v2_classifier"

# %%
# --- Step 1: Configure classifier ---
# NOTE: Update cell_type filter based on actual column values from Task 2
filter_data_dict = {"cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]}

training_args = {
    "num_train_epochs": 1,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "per_device_train_batch_size": 8,  # smaller batch for 316M model
    "seed": 42,
}

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    filter_data=filter_data_dict,
    training_args=training_args,
    max_ncells=None,
    freeze_layers=8,  # V2-316M has 12 layers; freeze most, fine-tune top 4
    num_crossval_splits=1,
    forward_batch_size=100,  # smaller for 316M
    model_version="V2",
    nproc=10,
    ngpu=1,
)

# %%
# --- Step 2: Prepare data with patient-level split ---
# NOTE: Update individual IDs based on actual data from Task 2
# These are the IDs from the paper's example; adjust for your data
# If using CELLxGENE data, the column may be "donor_id" instead of "individual"

# First, inspect available individuals:
from datasets import load_from_disk
ds = load_from_disk(INPUT_DATA)
individual_col = "individual"  # UPDATE if different
if individual_col in ds.column_names:
    individuals = sorted(set(ds[individual_col]))
    print(f"Found {len(individuals)} individuals: {individuals}")
else:
    print(f"Column '{individual_col}' not found. Available: {ds.column_names}")
    # Try alternatives
    for col in ["donor_id", "patient_id", "sample_id"]:
        if col in ds.column_names:
            individual_col = col
            individuals = sorted(set(ds[col]))
            print(f"Using '{col}' instead: {len(individuals)} individuals")
            break
del ds  # free memory

# Split: ~70% train, ~15% eval, ~15% test (by individual)
n = len(individuals)
n_train = int(n * 0.7)
n_eval = int(n * 0.15)
train_ids = individuals[:n_train]
eval_ids = individuals[n_train:n_train + n_eval]
test_ids = individuals[n_train + n_eval:]

print(f"Train: {len(train_ids)} individuals")
print(f"Eval: {len(eval_ids)} individuals")
print(f"Test: {len(test_ids)} individuals")

train_test_split = {
    "attr_key": individual_col,
    "train": train_ids + eval_ids,
    "test": test_ids,
}

cc.prepare_data(
    input_data_file=INPUT_DATA,
    output_directory=OUTPUT_DIR,
    output_prefix=output_prefix,
    split_id_dict=train_test_split,
)
print("Data prepared!")

# %%
# --- Step 3: Train ---
train_valid_split = {
    "attr_key": individual_col,
    "train": train_ids,
    "eval": eval_ids,
}

all_metrics = cc.validate(
    model_directory=MODEL_DIR,
    prepared_input_data_file=f"{OUTPUT_DIR}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{OUTPUT_DIR}/{output_prefix}_id_class_dict.pkl",
    output_directory=OUTPUT_DIR,
    output_prefix=output_prefix,
    split_id_dict=train_valid_split,
)
print(f"\nTraining metrics: F1={all_metrics['macro_f1']:.3f}, Acc={all_metrics['acc']:.3f}")

# %%
# --- Step 4: Evaluate on test set ---
cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    forward_batch_size=100,
    nproc=10,
)

# Find saved model
saved_model = glob.glob(f"{OUTPUT_DIR}/*cellClassifier*/ksplit1/")[0]
print(f"Using saved model: {saved_model}")

test_metrics = cc_eval.evaluate_saved_model(
    model_directory=saved_model,
    id_class_dict_file=f"{OUTPUT_DIR}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{OUTPUT_DIR}/{output_prefix}_labeled_test.dataset",
    output_directory=OUTPUT_DIR,
    output_prefix=output_prefix,
)
print(f"\nTest metrics: F1={test_metrics['macro_f1']:.3f}, Acc={test_metrics['acc']:.3f}")
print(f"Confusion matrix:\n{test_metrics['conf_matrix']}")

# %%
# --- Step 5: Plot results ---
cc_eval.plot_conf_mat(
    conf_mat_dict={"Geneformer-V2-316M": test_metrics["conf_matrix"]},
    output_directory=OUTPUT_DIR,
    output_prefix=output_prefix,
    custom_class_order=["nf", "hcm", "dcm"],
)

cc_eval.plot_predictions(
    predictions_file=f"{OUTPUT_DIR}/{output_prefix}_pred_dict.pkl",
    id_class_dict_file=f"{OUTPUT_DIR}/{output_prefix}_id_class_dict.pkl",
    title="Cardiomyopathy Disease Classification (V2-316M)",
    output_directory=OUTPUT_DIR,
    output_prefix=output_prefix,
    custom_class_order=["nf", "hcm", "dcm"],
)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Phase 2B complete!")
```

**Step 2: Run classification**

```bash
conda activate geneformer
python scripts/cm_reproduction/05_disease_classification.py
```

Expected: ~1-2 hours on GPU. Target: 90%+ accuracy. Output: fine-tuned model checkpoint, confusion matrix, prediction plots.

**Step 3: Commit**

```bash
git add scripts/cm_reproduction/05_disease_classification.py
git commit -m "Add disease classification script (Phase 2B)"
```

---

## Task 8: In Silico Treatment Analysis (Phase 2C)

**Files:**
- Create: `scripts/cm_reproduction/06_treatment_analysis.py`

**Depends on:** Task 7 (fine-tuned classifier model)

**Step 1: Write treatment analysis script**

Create `scripts/cm_reproduction/06_treatment_analysis.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Phase 2C: In silico treatment analysis.
Replicates Fig. 6c-e from Theodoris et al. 2023.

Finds genes whose deletion shifts diseased cardiomyocytes toward
non-failing state. Runs two directions: HCM->NF and DCM->NF.

Uses fine-tuned classifier from Phase 2B.
"""
import matplotlib
matplotlib.use("Agg")
import datetime
import glob
import os
import pickle
from geneformer import EmbExtractor, InSilicoPerturber, InSilicoPerturberStats

# --- Paths ---
INPUT_DATA = "/home/jw3514/Work/Geneformer/Geneformer/data/tokenized/chaffin_cm.dataset"
GENE_LIST_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/gene_lists"

# Find latest fine-tuned model from Phase 2B
phase2b_dirs = sorted(glob.glob(
    "/home/jw3514/Work/Geneformer/Geneformer/outputs/phase2b_classification/*/"))
if not phase2b_dirs:
    raise FileNotFoundError("No Phase 2B output found. Run 05_disease_classification.py first.")
latest_phase2b = phase2b_dirs[-1]
model_candidates = glob.glob(f"{latest_phase2b}/*cellClassifier*/ksplit1/")
if not model_candidates:
    raise FileNotFoundError(f"No fine-tuned model in {latest_phase2b}")
FINETUNED_MODEL = model_candidates[0]
print(f"Using fine-tuned model: {FINETUNED_MODEL}")

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
OUTPUT_DIR = f"/home/jw3514/Work/Geneformer/Geneformer/outputs/phase2c_treatment_analysis/{datestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cell type filter (must match Phase 2B)
filter_data_dict = {"cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]}

# Optionally, restrict to TFs only for faster initial results
with open(os.path.join(GENE_LIST_DIR, "cardiac_tfs.pkl"), "rb") as f:
    cardiac_tfs = pickle.load(f)
print(f"Loaded {len(cardiac_tfs)} cardiac TFs for perturbation")

# Set to cardiac_tfs for fast run, "all" for full reproduction
GENES_TO_PERTURB = cardiac_tfs  # Change to "all" for full analysis

# %%
# === Direction 1: DCM -> NF ===
print("=" * 60)
print("Direction 1: DCM -> NF")
print("=" * 60)

dcm_states = {
    "state_key": "disease",
    "start_state": "dcm",
    "goal_state": "nf",
    "alt_states": ["hcm"],
}

# Step 1: Extract state embeddings
embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=3,
    filter_data=filter_data_dict,
    max_ncells=2000,
    emb_layer=0,  # last layer (task-specific, matching existing scripts)
    summary_stat="exact_mean",
    forward_batch_size=100,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    model_version="V2",
    nproc=10,
)

dcm_state_embs = embex.get_state_embs(
    dcm_states,
    FINETUNED_MODEL,
    INPUT_DATA,
    OUTPUT_DIR,
    "dcm_to_nf_state_embs",
)
print("DCM state embeddings extracted!")

# Step 2: In silico perturbation
isp_dcm = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=GENES_TO_PERTURB,
    combos=0,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data=filter_data_dict,
    cell_states_to_model=dcm_states,
    state_embs_dict=dcm_state_embs,
    max_ncells=2000,
    emb_layer=0,
    forward_batch_size=100,
    model_version="V2",
    nproc=10,
)

isp_dcm.perturb_data(
    FINETUNED_MODEL,
    INPUT_DATA,
    OUTPUT_DIR,
    "dcm_to_nf",
)
print("DCM perturbation complete!")

# Step 3: Statistics
ispstats_dcm = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed=GENES_TO_PERTURB if GENES_TO_PERTURB != "all" else "all",
    combos=0,
    cell_states_to_model=dcm_states,
    model_version="V2",
)

ispstats_dcm.get_stats(
    OUTPUT_DIR,
    None,
    OUTPUT_DIR,
    "dcm_to_nf",
)
print("DCM stats complete!")

# %%
# === Direction 2: HCM -> NF ===
print("=" * 60)
print("Direction 2: HCM -> NF")
print("=" * 60)

hcm_states = {
    "state_key": "disease",
    "start_state": "hcm",
    "goal_state": "nf",
    "alt_states": ["dcm"],
}

hcm_state_embs = embex.get_state_embs(
    hcm_states,
    FINETUNED_MODEL,
    INPUT_DATA,
    OUTPUT_DIR,
    "hcm_to_nf_state_embs",
)
print("HCM state embeddings extracted!")

isp_hcm = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=GENES_TO_PERTURB,
    combos=0,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data=filter_data_dict,
    cell_states_to_model=hcm_states,
    state_embs_dict=hcm_state_embs,
    max_ncells=2000,
    emb_layer=0,
    forward_batch_size=100,
    model_version="V2",
    nproc=10,
)

isp_hcm.perturb_data(
    FINETUNED_MODEL,
    INPUT_DATA,
    OUTPUT_DIR,
    "hcm_to_nf",
)
print("HCM perturbation complete!")

ispstats_hcm = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed=GENES_TO_PERTURB if GENES_TO_PERTURB != "all" else "all",
    combos=0,
    cell_states_to_model=hcm_states,
    model_version="V2",
)

ispstats_hcm.get_stats(
    OUTPUT_DIR,
    None,
    OUTPUT_DIR,
    "hcm_to_nf",
)
print("HCM stats complete!")

# %%
# === Analyze results ===
import pandas as pd

for direction in ["dcm_to_nf", "hcm_to_nf"]:
    stats_file = os.path.join(OUTPUT_DIR, f"{direction}.csv")
    if os.path.exists(stats_file):
        df = pd.read_csv(stats_file)
        print(f"\n{'='*60}")
        print(f"{direction.upper()} Results:")
        print(f"Total genes analyzed: {len(df)}")
        sig = df[df.get("Sig", df.get("Goal_end_FDR", pd.Series())) == 1] if "Sig" in df.columns else df[df["Goal_end_FDR"] < 0.05]
        print(f"Significant genes (FDR < 0.05): {len(sig)}")
        print(f"\nTop 15 genes shifting toward NF:")
        sort_col = "Shift_to_goal_end" if "Shift_to_goal_end" in df.columns else df.columns[-1]
        print(df.nlargest(15, sort_col)[["Gene_name", sort_col]].to_string(index=False))
    else:
        print(f"Stats file not found: {stats_file}")

# %%
# === Save combined top TFs for Phase 2D ===
dcm_stats = os.path.join(OUTPUT_DIR, "dcm_to_nf.csv")
hcm_stats = os.path.join(OUTPUT_DIR, "hcm_to_nf.csv")

if os.path.exists(dcm_stats) and os.path.exists(hcm_stats):
    df_dcm = pd.read_csv(dcm_stats)
    df_hcm = pd.read_csv(hcm_stats)

    # Combine: take top 50 TFs by shift from each direction
    sort_col = "Shift_to_goal_end"
    top_dcm = set(df_dcm.nlargest(50, sort_col)["Ensembl_ID"].tolist())
    top_hcm = set(df_hcm.nlargest(50, sort_col)["Ensembl_ID"].tolist())
    top_tfs_combined = list(top_dcm | top_hcm)

    with open(os.path.join(OUTPUT_DIR, "top_tfs_for_phase2d.pkl"), "wb") as f:
        pickle.dump(top_tfs_combined, f)
    print(f"\nSaved {len(top_tfs_combined)} top TFs for Phase 2D")

    # Overlap analysis
    overlap = top_dcm & top_hcm
    print(f"Overlap between DCM and HCM top TFs: {len(overlap)}")

print("\nPhase 2C complete!")
```

**Step 2: Run treatment analysis**

```bash
conda activate geneformer
python scripts/cm_reproduction/06_treatment_analysis.py
```

Expected: Long runtime (~4-24h depending on gene count). With TFs only (~50 genes): ~1-2h. Output: CSVs ranking genes by shift toward NF state.

**Step 3: Commit**

```bash
git add scripts/cm_reproduction/06_treatment_analysis.py
git commit -m "Add in silico treatment analysis script (Phase 2C)"
```

---

## Task 9: Per-Gene Expression Impact Extraction (Phase 2D)

**Files:**
- Create: `scripts/cm_reproduction/07_per_gene_impact.py`

**Depends on:** Task 8 (top TFs identified, state embeddings)

**Step 1: Write per-gene impact script**

Create `scripts/cm_reproduction/07_per_gene_impact.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Phase 2D: Per-gene expression impact extraction.

For top TFs from Phase 2C, re-run perturbation with emb_mode="cell_and_gene"
to get per-gene embedding shifts. Filter for ion channel genes to prepare
for future openCARP integration.
"""
import matplotlib
matplotlib.use("Agg")
import glob
import os
import pickle
import pandas as pd
import numpy as np
from geneformer import EmbExtractor, InSilicoPerturber
from geneformer import ENSEMBL_DICTIONARY_FILE

# --- Paths ---
INPUT_DATA = "/home/jw3514/Work/Geneformer/Geneformer/data/tokenized/chaffin_cm.dataset"
GENE_LIST_DIR = "/home/jw3514/Work/Geneformer/Geneformer/data/gene_lists"

# Find latest Phase 2C output
phase2c_dirs = sorted(glob.glob(
    "/home/jw3514/Work/Geneformer/Geneformer/outputs/phase2c_treatment_analysis/*/"))
if not phase2c_dirs:
    raise FileNotFoundError("No Phase 2C output found. Run 06_treatment_analysis.py first.")
latest_phase2c = phase2c_dirs[-1]

# Load top TFs from Phase 2C
top_tfs_file = os.path.join(latest_phase2c, "top_tfs_for_phase2d.pkl")
with open(top_tfs_file, "rb") as f:
    top_tfs = pickle.load(f)
print(f"Loaded {len(top_tfs)} top TFs for perturbation")

# Load ion channel gene list
with open(os.path.join(GENE_LIST_DIR, "ion_channel_genes.pkl"), "rb") as f:
    ion_channel_genes = pickle.load(f)

# Load gene name mapping
with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
    gene_name_to_id = pickle.load(f)
gene_id_to_name = {v: k for k, v in gene_name_to_id.items()}

# Find fine-tuned model (from Phase 2B)
phase2b_dirs = sorted(glob.glob(
    "/home/jw3514/Work/Geneformer/Geneformer/outputs/phase2b_classification/*/"))
FINETUNED_MODEL = glob.glob(f"{phase2b_dirs[-1]}/*cellClassifier*/ksplit1/")[0]
print(f"Using fine-tuned model: {FINETUNED_MODEL}")

OUTPUT_DIR = "/home/jw3514/Work/Geneformer/Geneformer/outputs/phase2d_per_gene_impact"
os.makedirs(OUTPUT_DIR, exist_ok=True)

filter_data_dict = {"cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]}

# %%
# --- Step 1: Extract state embeddings (reuse from 2C or re-extract) ---
dcm_states = {
    "state_key": "disease",
    "start_state": "dcm",
    "goal_state": "nf",
    "alt_states": ["hcm"],
}

embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=3,
    filter_data=filter_data_dict,
    max_ncells=2000,
    emb_layer=0,
    summary_stat="exact_mean",
    forward_batch_size=100,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    model_version="V2",
    nproc=10,
)

state_embs = embex.get_state_embs(
    dcm_states,
    FINETUNED_MODEL,
    INPUT_DATA,
    OUTPUT_DIR,
    "state_embs_2d",
)

# %%
# --- Step 2: Run perturbation with cell_and_gene mode ---
print("Running per-gene perturbation for top TFs...")
print("NOTE: This uses emb_mode='cell_and_gene' which outputs per-gene shifts")
print(f"Perturbing {len(top_tfs)} TFs")

isp = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=top_tfs,
    combos=0,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cell_and_gene",  # KEY: get per-gene embedding shifts
    cell_emb_style="mean_pool",
    filter_data=filter_data_dict,
    cell_states_to_model=dcm_states,
    state_embs_dict=state_embs,
    max_ncells=500,  # smaller to manage memory with gene-level output
    emb_layer=0,
    forward_batch_size=50,  # smaller batch for gene-level analysis
    model_version="V2",
    nproc=10,
)

isp.perturb_data(
    FINETUNED_MODEL,
    INPUT_DATA,
    OUTPUT_DIR,
    "per_gene_impact",
)
print("Per-gene perturbation complete!")

# %%
# --- Step 3: Extract ion channel shifts from gene-level output ---
print("\nExtracting ion channel gene shifts...")

# Load gene-level perturbation results
gene_embs_files = glob.glob(os.path.join(OUTPUT_DIR, "*gene_embs_dict*_raw.pickle"))
if not gene_embs_files:
    print("WARNING: No gene_embs_dict files found. Check output directory.")
else:
    # Process gene-level results
    results = []
    for gf in gene_embs_files:
        with open(gf, "rb") as f:
            gene_embs_dict = pickle.load(f)

        # gene_embs_dict structure: {perturbed_gene_token: {target_gene_token: [cosine_sims]}}
        for perturbed_token, target_dict in gene_embs_dict.items():
            perturbed_name = gene_id_to_name.get(perturbed_token, str(perturbed_token))
            for target_token, cos_sims in target_dict.items():
                target_ensembl = str(target_token)
                # Check if this target is an ion channel gene
                if target_ensembl in ion_channel_genes or target_token in ion_channel_genes:
                    target_name = gene_id_to_name.get(target_token, target_ensembl)
                    avg_shift = np.mean(cos_sims) if isinstance(cos_sims, list) else cos_sims
                    results.append({
                        "TF_deleted": perturbed_name,
                        "TF_ensembl": str(perturbed_token),
                        "ion_channel_gene": target_name,
                        "ion_channel_ensembl": target_ensembl,
                        "mean_cosine_shift": avg_shift,
                        "n_cells": len(cos_sims) if isinstance(cos_sims, list) else 1,
                    })

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values("mean_cosine_shift", ascending=True)
        outpath = os.path.join(OUTPUT_DIR, "tf_ion_channel_shifts.csv")
        df_results.to_csv(outpath, index=False)
        print(f"\nSaved {len(df_results)} TF x ion channel entries to {outpath}")
        print(f"\nTop TF-ion channel interactions (most deleterious):")
        print(df_results.head(20).to_string(index=False))
    else:
        print("No ion channel genes found in perturbation output.")
        print("This may be because the gene token IDs need different mapping.")
        print("Check the gene_embs_dict keys vs ion_channel_genes Ensembl IDs.")

print("\nPhase 2D complete!")
```

**Step 2: Run per-gene impact extraction**

```bash
conda activate geneformer
python scripts/cm_reproduction/07_per_gene_impact.py
```

Expected: ~2-4 hours. Output: `tf_ion_channel_shifts.csv` with per-TF x per-ion-channel cosine shifts.

**Step 3: Commit**

```bash
git add scripts/cm_reproduction/07_per_gene_impact.py
git commit -m "Add per-gene impact extraction script (Phase 2D)"
```

---

## Task 10: Summary Analysis and Visualization

**Files:**
- Create: `scripts/cm_reproduction/08_summary.py`

**Depends on:** Tasks 6-9 (all phases complete)

**Step 1: Write summary script**

Create `scripts/cm_reproduction/08_summary.py`:

```python
# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Summary analysis: combine results from all phases.
Generate publication-quality figures and summary tables.
"""
import matplotlib
matplotlib.use("Agg")
import glob
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geneformer import ENSEMBL_DICTIONARY_FILE

OUTPUT_DIR = "/home/jw3514/Work/Geneformer/Geneformer/outputs"

with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
    gene_name_to_id = pickle.load(f)
gene_id_to_name = {v: k for k, v in gene_name_to_id.items()}

# %%
# --- Phase 2C: Treatment analysis summary ---
phase2c_dirs = sorted(glob.glob(f"{OUTPUT_DIR}/phase2c_treatment_analysis/*/"))
if phase2c_dirs:
    latest = phase2c_dirs[-1]
    for direction in ["dcm_to_nf", "hcm_to_nf"]:
        csv_path = os.path.join(latest, f"{direction}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sort_col = "Shift_to_goal_end"
            if sort_col in df.columns:
                print(f"\n{'='*60}")
                print(f"{direction.upper()}: Top 25 therapeutic TF targets")
                print(f"{'='*60}")
                top = df.nlargest(25, sort_col)
                print(top[["Gene_name", sort_col, "Goal_end_FDR", "N_Detections"]].to_string(index=False))

    # Overlap Venn-like analysis
    dcm_csv = os.path.join(latest, "dcm_to_nf.csv")
    hcm_csv = os.path.join(latest, "hcm_to_nf.csv")
    if os.path.exists(dcm_csv) and os.path.exists(hcm_csv):
        df_dcm = pd.read_csv(dcm_csv)
        df_hcm = pd.read_csv(hcm_csv)
        sort_col = "Shift_to_goal_end"

        sig_dcm = set(df_dcm[df_dcm["Goal_end_FDR"] < 0.05]["Gene_name"].tolist())
        sig_hcm = set(df_hcm[df_hcm["Goal_end_FDR"] < 0.05]["Gene_name"].tolist())
        overlap = sig_dcm & sig_hcm

        print(f"\n{'='*60}")
        print(f"Overlap Analysis (FDR < 0.05)")
        print(f"DCM-specific: {len(sig_dcm - overlap)}")
        print(f"HCM-specific: {len(sig_hcm - overlap)}")
        print(f"Shared: {len(overlap)}")
        if overlap:
            print(f"Shared genes: {sorted(overlap)[:20]}...")

# %%
# --- Phase 2D: Ion channel impact summary ---
ic_csv = os.path.join(OUTPUT_DIR, "phase2d_per_gene_impact", "tf_ion_channel_shifts.csv")
if os.path.exists(ic_csv):
    df_ic = pd.read_csv(ic_csv)
    print(f"\n{'='*60}")
    print(f"TF -> Ion Channel Impact Summary")
    print(f"{'='*60}")
    print(f"Total entries: {len(df_ic)}")
    print(f"Unique TFs: {df_ic['TF_deleted'].nunique()}")
    print(f"Ion channels analyzed: {df_ic['ion_channel_gene'].unique().tolist()}")

    # Pivot table: TF x ion channel
    pivot = df_ic.pivot_table(
        index="TF_deleted",
        columns="ion_channel_gene",
        values="mean_cosine_shift",
        aggfunc="mean",
    )
    print(f"\nTF x Ion Channel shift matrix:")
    print(pivot.to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    sns.heatmap(pivot, cmap="RdBu_r", center=0, annot=True, fmt=".3f", ax=ax)
    ax.set_title("TF Deletion Impact on Ion Channel Gene Embeddings")
    ax.set_ylabel("Deleted TF")
    ax.set_xlabel("Ion Channel Gene")
    fig.savefig(os.path.join(OUTPUT_DIR, "phase2d_per_gene_impact", "tf_ion_channel_heatmap.png"),
                dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print("\nHeatmap saved!")

print("\nSummary analysis complete!")
```

**Step 2: Run summary**

```bash
conda activate geneformer
python scripts/cm_reproduction/08_summary.py
```

**Step 3: Commit all scripts**

```bash
git add scripts/cm_reproduction/08_summary.py
git commit -m "Add summary analysis and visualization script"
```

---

## Important Notes

### Data Acquisition Fallback

If downloading raw Chaffin et al. data proves difficult (dbGaP authorization, Broad SCP registration), the **pre-tokenized V1 dataset** (`human_dcm_hcm_nf.dataset`) can be used as a fallback for Tasks 7-9. The V1 tokenization uses different gene medians, so V2 model predictions may be slightly suboptimal, but the pipeline will still work. Skip Task 4 tokenization for adult data and update paths in Tasks 7-9 accordingly.

### Runtime Estimates

| Task | GPU Time | CPU Time |
|------|----------|----------|
| Task 4: Tokenization | N/A | ~30 min |
| Task 6: Zero-shot deletion | ~2-4h | ~30 min |
| Task 7: Classification | ~1-2h | ~10 min |
| Task 8: Treatment analysis (TFs only) | ~2-4h per direction | ~30 min |
| Task 8: Treatment analysis (all genes) | ~12-24h per direction | ~1h |
| Task 9: Per-gene impact | ~2-4h | ~30 min |

### Column Name Updates

Tasks 4, 6, 7, 8 contain `# UPDATE` comments where column names must be adjusted based on actual data inspection in Tasks 2-3. The exact column names for `disease`, `cell_type`, and `individual` depend on the data source used.

### Memory Management for V2-316M

If GPU OOM errors occur:
- Reduce `forward_batch_size` (try 50, then 25)
- Reduce `max_ncells` (try 1000, then 500)
- Consider using quantization: add `quantize=True` to Classifier constructor
- Use `emb_layer=-1` instead of `0` (same memory but sometimes more stable)
