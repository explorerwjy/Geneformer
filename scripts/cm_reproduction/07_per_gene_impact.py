# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Per-Gene Impact Extraction (Phase 2D)
#
# Uses `emb_mode="cls_and_gene"` to measure how deleting top therapeutic TFs
# (from Phase 2C) affects individual gene embeddings, specifically focusing on
# ion channel genes.
#
# ## Approach
# 1. Load top TFs identified in Phase 2C.
# 2. Re-run InSilicoPerturber with `emb_mode="cls_and_gene"` to capture
#    both cell-level and gene-level embedding shifts.
# 3. Extract shifts for ion channel genes specifically.
# 4. Build a TF x ion channel shift matrix and save as CSV.

# %%
import glob
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geneformer import EmbExtractor, InSilicoPerturber, InSilicoPerturberStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# === Paths ===
BASE_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer")
MODEL_DIR = BASE_DIR / "models" / "Geneformer" / "Geneformer-V2-316M"
INPUT_DATA = BASE_DIR / "data" / "tokenized" / "chaffin_cardiomyocytes.dataset"
GENE_LIST_DIR = BASE_DIR / "data" / "gene_lists"

# Phase 2C outputs
TREATMENT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "treatment_analysis"
STATE_EMBS_DIR = TREATMENT_DIR / "state_embs"
STATS_DIR = TREATMENT_DIR / "stats"

# Phase 2D outputs
OUTPUT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "per_gene_impact"
ISP_OUTPUT_DIR_DCM = OUTPUT_DIR / "isp_dcm_gene"
ISP_OUTPUT_DIR_HCM = OUTPUT_DIR / "isp_hcm_gene"
GENE_STATS_DIR = OUTPUT_DIR / "gene_stats"

for d in [ISP_OUTPUT_DIR_DCM, ISP_OUTPUT_DIR_HCM, GENE_STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# %%
# === Load gene lists ===
with open(GENE_LIST_DIR / "cardiac_tfs.pkl", "rb") as f:
    cardiac_tfs = pickle.load(f)

with open(GENE_LIST_DIR / "ion_channel_genes.pkl", "rb") as f:
    ion_channel_genes = pickle.load(f)

print(f"Cardiac TFs: {len(cardiac_tfs)}")
print(f"Ion channel genes: {len(ion_channel_genes)}")

# %%
# === Load top TFs from Phase 2C ===
top_tfs_dcm_file = STATS_DIR / "top_tfs_dcm_to_nf.pkl"
top_tfs_hcm_file = STATS_DIR / "top_tfs_hcm_to_nf.pkl"

if not top_tfs_dcm_file.exists() or not top_tfs_hcm_file.exists():
    print("=" * 60)
    print("ERROR: Top TF files from Phase 2C not found.")
    print(f"Expected: {top_tfs_dcm_file}")
    print(f"Expected: {top_tfs_hcm_file}")
    print("Please run script 06_treatment_analysis.py first.")
    print("=" * 60)
    sys.exit(1)

with open(top_tfs_dcm_file, "rb") as f:
    top_tfs_dcm = pickle.load(f)

with open(top_tfs_hcm_file, "rb") as f:
    top_tfs_hcm = pickle.load(f)

print(f"Top DCM->NF TFs: {len(top_tfs_dcm)}")
print(f"Top HCM->NF TFs: {len(top_tfs_hcm)}")

# %%
# === Load state embeddings from Phase 2C ===
state_embs_file = STATE_EMBS_DIR / "disease_state_embs.pkl"
if not state_embs_file.exists():
    print("=" * 60)
    print("ERROR: State embeddings not found.")
    print(f"Expected: {state_embs_file}")
    print("Please run script 06_treatment_analysis.py first.")
    print("=" * 60)
    sys.exit(1)

with open(state_embs_file, "rb") as f:
    state_embs_dict = pickle.load(f)

print(f"State embeddings loaded for: {list(state_embs_dict.keys())}")

# %%
# === Find fine-tuned classifier model ===
CLASSIFIER_BASE = BASE_DIR / "outputs" / "disease_classification"
model_candidates = sorted(glob.glob(str(CLASSIFIER_BASE / "*" / "*cellClassifier*" / "ksplit1")))

if not model_candidates:
    print("=" * 60)
    print("ERROR: No fine-tuned classifier model found.")
    print("Please run script 05_disease_classification.py first.")
    print("=" * 60)
    sys.exit(1)

FINETUNED_MODEL = model_candidates[-1]
print(f"Using fine-tuned model: {FINETUNED_MODEL}")

# %%
# === Configuration ===
MAX_NCELLS = 2000  # Set to 50 for quick testing, 2000 for full run
FORWARD_BATCH_SIZE = 100

dcm_states = {
    "state_key": "disease",
    "start_state": "DCM",
    "goal_state": "NF",
    "alt_states": ["HCM"],
}

hcm_states = {
    "state_key": "disease",
    "start_state": "HCM",
    "goal_state": "NF",
    "alt_states": ["DCM"],
}

# %% [markdown]
# ## Step 1: Per-gene perturbation - DCM to NF (top TFs)

# %%
print("=" * 60)
print("Running per-gene perturbation: DCM -> NF (cls_and_gene mode)")
print("=" * 60)

isp_dcm_gene = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=top_tfs_dcm,
    combos=0,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cls_and_gene",
    cell_emb_style="mean_pool",
    filter_data={"disease": ["DCM"]},
    cell_states_to_model=dcm_states,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=0,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version="V2",
    nproc=10,
)

isp_dcm_gene.perturb_data(
    model_directory=FINETUNED_MODEL,
    input_data_file=str(INPUT_DATA),
    output_directory=str(ISP_OUTPUT_DIR_DCM),
    output_prefix="dcm_gene",
)

# %% [markdown]
# ## Step 2: Per-gene perturbation - HCM to NF (top TFs)

# %%
print("=" * 60)
print("Running per-gene perturbation: HCM -> NF (cls_and_gene mode)")
print("=" * 60)

isp_hcm_gene = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=top_tfs_hcm,
    combos=0,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cls_and_gene",
    cell_emb_style="mean_pool",
    filter_data={"disease": ["HCM"]},
    cell_states_to_model=hcm_states,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=0,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version="V2",
    nproc=10,
)

isp_hcm_gene.perturb_data(
    model_directory=FINETUNED_MODEL,
    input_data_file=str(INPUT_DATA),
    output_directory=str(ISP_OUTPUT_DIR_HCM),
    output_prefix="hcm_gene",
)

# %% [markdown]
# ## Step 3: Aggregate gene-level shifts

# %%
print("=" * 60)
print("Aggregating gene shifts: DCM -> NF")
print("=" * 60)

ispstats_dcm_gene = InSilicoPerturberStats(
    mode="aggregate_gene_shifts",
    genes_perturbed="all",
    combos=0,
    cell_states_to_model=dcm_states,
    model_version="V2",
)

ispstats_dcm_gene.get_stats(
    input_data_directory=str(ISP_OUTPUT_DIR_DCM),
    null_dist_data_directory=None,
    output_directory=str(GENE_STATS_DIR),
    output_prefix="dcm_gene_shifts",
)

# %%
print("=" * 60)
print("Aggregating gene shifts: HCM -> NF")
print("=" * 60)

ispstats_hcm_gene = InSilicoPerturberStats(
    mode="aggregate_gene_shifts",
    genes_perturbed="all",
    combos=0,
    cell_states_to_model=hcm_states,
    model_version="V2",
)

ispstats_hcm_gene.get_stats(
    input_data_directory=str(ISP_OUTPUT_DIR_HCM),
    null_dist_data_directory=None,
    output_directory=str(GENE_STATS_DIR),
    output_prefix="hcm_gene_shifts",
)

# %% [markdown]
# ## Step 4: Extract ion channel shifts and build TF x ion channel matrix

# %%
# Load the gene-level shift results
# The output format from aggregate_gene_shifts is CSV with per-gene cosine shifts
dcm_gene_files = sorted(GENE_STATS_DIR.glob("dcm_gene_shifts*.csv"))
hcm_gene_files = sorted(GENE_STATS_DIR.glob("hcm_gene_shifts*.csv"))

print(f"DCM gene shift files: {dcm_gene_files}")
print(f"HCM gene shift files: {hcm_gene_files}")

# %%
# Load token dictionary for gene name mapping
from geneformer import TOKEN_DICTIONARY_FILE, ENSEMBL_DICTIONARY_FILE

with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dict = pickle.load(f)
gene_token_dict = {v: k for k, v in token_dict.items()}

with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
    ensembl_dict = pickle.load(f)
token_to_ensembl = {v: k for k, v in ensembl_dict.items()}

# Map ion channel Ensembl IDs to gene names for display
ion_channel_names = {}
for ens_id in ion_channel_genes:
    if ens_id in ensembl_dict:
        ion_channel_names[ens_id] = ensembl_dict[ens_id]
    else:
        ion_channel_names[ens_id] = ens_id

print(f"Ion channel gene mapping:")
for ens_id, name in ion_channel_names.items():
    print(f"  {ens_id} -> {name}")

# %%
# Build TF x ion channel shift matrices
# The exact format depends on InSilicoPerturberStats output; we handle common formats
def build_shift_matrix(gene_shift_files, tfs_list, ion_channels, ensembl_dict):
    """Build a TF x ion channel shift matrix from gene shift CSV files."""
    if not gene_shift_files:
        print("No gene shift files found.")
        return None

    # Try to load and parse the gene shift data
    all_data = []
    for f in gene_shift_files:
        df = pd.read_csv(f)
        all_data.append(df)
        print(f"Loaded {f}: shape {df.shape}, columns {df.columns.tolist()}")

    combined = pd.concat(all_data, ignore_index=True)

    # The output may have columns like Gene, Ensembl_ID, shift values
    # Try to pivot into TF x target gene matrix
    print(f"\nCombined shape: {combined.shape}")
    print(f"Sample rows:\n{combined.head(10).to_string()}")

    return combined


dcm_shifts = build_shift_matrix(dcm_gene_files, top_tfs_dcm, ion_channel_genes, ensembl_dict)
hcm_shifts = build_shift_matrix(hcm_gene_files, top_tfs_hcm, ion_channel_genes, ensembl_dict)

# %%
# Save results
if dcm_shifts is not None:
    dcm_shifts.to_csv(GENE_STATS_DIR / "dcm_tf_ion_channel_shifts.csv", index=False)
    print(f"Saved DCM TF x ion channel shifts")

if hcm_shifts is not None:
    hcm_shifts.to_csv(GENE_STATS_DIR / "hcm_tf_ion_channel_shifts.csv", index=False)
    print(f"Saved HCM TF x ion channel shifts")

# %%
print("=" * 60)
print("PER-GENE IMPACT EXTRACTION COMPLETE (Phase 2D)")
print("=" * 60)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  DCM gene perturbation: {ISP_OUTPUT_DIR_DCM}")
print(f"  HCM gene perturbation: {ISP_OUTPUT_DIR_HCM}")
print(f"  Gene shift statistics: {GENE_STATS_DIR}")
