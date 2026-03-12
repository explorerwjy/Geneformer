# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # In Silico Treatment Analysis (Phase 2C)
#
# Uses the fine-tuned disease classifier from Phase 2B to identify transcription
# factors whose deletion shifts diseased cardiomyocyte embeddings toward the
# non-failing (NF) state.
#
# ## Approach
# 1. Extract state embeddings (NF, HCM, DCM centroids) from the fine-tuned model.
# 2. Run InSilicoPerturber to delete cardiac TFs in DCM and HCM cells.
# 3. Compute goal_state_shift statistics to rank TFs by their ability to shift
#    diseased cells toward the NF state.
# 4. Save top TFs for downstream per-gene impact analysis (Phase 2D).

# %%
import glob
import logging
import os
import pickle
import sys
from pathlib import Path

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

OUTPUT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "treatment_analysis"
STATE_EMBS_DIR = OUTPUT_DIR / "state_embs"
ISP_OUTPUT_DIR_DCM = OUTPUT_DIR / "isp_dcm_to_nf"
ISP_OUTPUT_DIR_HCM = OUTPUT_DIR / "isp_hcm_to_nf"
STATS_OUTPUT_DIR = OUTPUT_DIR / "stats"

for d in [STATE_EMBS_DIR, ISP_OUTPUT_DIR_DCM, ISP_OUTPUT_DIR_HCM, STATS_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# %%
# === Load gene lists ===
with open(GENE_LIST_DIR / "cardiac_tfs.pkl", "rb") as f:
    cardiac_tfs = pickle.load(f)

print(f"Cardiac TFs: {len(cardiac_tfs)}")

# %%
# === Find fine-tuned classifier model ===
# The classifier is trained in Phase 2B (script 05). Look for the most recent
# output directory containing a cellClassifier model.
CLASSIFIER_BASE = BASE_DIR / "outputs" / "disease_classification"

if not CLASSIFIER_BASE.exists():
    print("=" * 60)
    print("ERROR: No disease_classification output directory found.")
    print(f"Expected at: {CLASSIFIER_BASE}")
    print("Please run script 05_disease_classification.py first.")
    print("=" * 60)
    sys.exit(1)

# Find the most recent datestamp directory containing a trained model
model_candidates = sorted(glob.glob(str(CLASSIFIER_BASE / "*" / "*cellClassifier*" / "ksplit1")))

if not model_candidates:
    print("=" * 60)
    print("ERROR: No fine-tuned classifier model found.")
    print(f"Searched in: {CLASSIFIER_BASE}")
    print("Please run script 05_disease_classification.py first to train the model.")
    print("=" * 60)
    sys.exit(1)

FINETUNED_MODEL = model_candidates[-1]  # most recent
print(f"Using fine-tuned model: {FINETUNED_MODEL}")

# %% [markdown]
# ## Step 1: Extract state embeddings (NF, HCM, DCM centroids)

# %%
# Configuration shared between DCM->NF and HCM->NF analyses
MAX_NCELLS = 2000  # Set to 50 for quick testing, 2000 for full run
FORWARD_BATCH_SIZE = 50  # CellClassifier model uses more VRAM than pretrained
EMB_BATCH_SIZE = 50  # EmbExtractor uses classifier model (more VRAM than pretrained)

# Cell states dictionaries for the two directions
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

# %%
# Extract state embeddings using the fine-tuned classifier
embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=3,
    filter_data=None,
    max_ncells=MAX_NCELLS,
    emb_layer=0,
    summary_stat="exact_mean",
    forward_batch_size=EMB_BATCH_SIZE,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    model_version="V2",
    nproc=1,  # use 1 to avoid multiprocessing spawn issues
)

# Extract state embeddings (same for both directions since we model all 3 states)
state_embs_dict = embex.get_state_embs(
    cell_states_to_model=dcm_states,
    model_directory=FINETUNED_MODEL,
    input_data_file=str(INPUT_DATA),
    output_directory=str(STATE_EMBS_DIR),
    output_prefix="disease_state_embs",
)

print(f"State embeddings extracted for: {list(state_embs_dict.keys())}")
for state, emb in state_embs_dict.items():
    print(f"  {state}: shape {emb.shape}")

# %% [markdown]
# ## Step 2: In silico perturbation - DCM to NF

# %%
print("=" * 60)
print("Running in silico perturbation: DCM -> NF")
print("=" * 60)

isp_dcm = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb="all",
    combos=0,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    filter_data={"disease": ["DCM"]},
    cell_states_to_model=dcm_states,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=0,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version="V2",
    nproc=1,  # use 1 to avoid multiprocessing spawn issues
)

isp_dcm.perturb_data(
    model_directory=FINETUNED_MODEL,
    input_data_file=str(INPUT_DATA),
    output_directory=str(ISP_OUTPUT_DIR_DCM),
    output_prefix="dcm_to_nf",
)

# %% [markdown]
# ## Step 3: In silico perturbation - HCM to NF

# %%
print("=" * 60)
print("Running in silico perturbation: HCM -> NF")
print("=" * 60)

isp_hcm = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb="all",
    combos=0,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    filter_data={"disease": ["HCM"]},
    cell_states_to_model=hcm_states,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=0,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version="V2",
    nproc=1,  # use 1 to avoid multiprocessing spawn issues
)

isp_hcm.perturb_data(
    model_directory=FINETUNED_MODEL,
    input_data_file=str(INPUT_DATA),
    output_directory=str(ISP_OUTPUT_DIR_HCM),
    output_prefix="hcm_to_nf",
)

# %% [markdown]
# ## Step 4: Compute goal state shift statistics

# %%
print("=" * 60)
print("Computing goal_state_shift statistics: DCM -> NF")
print("=" * 60)

ispstats_dcm = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed="all",
    combos=0,
    cell_states_to_model=dcm_states,
    model_version="V2",
)

ispstats_dcm.get_stats(
    input_data_directory=str(ISP_OUTPUT_DIR_DCM),
    null_dist_data_directory=None,
    output_directory=str(STATS_OUTPUT_DIR),
    output_prefix="dcm_to_nf_stats",
)

# %%
print("=" * 60)
print("Computing goal_state_shift statistics: HCM -> NF")
print("=" * 60)

ispstats_hcm = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed="all",
    combos=0,
    cell_states_to_model=hcm_states,
    model_version="V2",
)

ispstats_hcm.get_stats(
    input_data_directory=str(ISP_OUTPUT_DIR_HCM),
    null_dist_data_directory=None,
    output_directory=str(STATS_OUTPUT_DIR),
    output_prefix="hcm_to_nf_stats",
)

# %% [markdown]
# ## Step 5: Load and inspect results

# %%
# Load DCM->NF results
dcm_stats_files = sorted(STATS_OUTPUT_DIR.glob("dcm_to_nf_stats*.csv"))
print(f"DCM->NF stats files: {dcm_stats_files}")

if dcm_stats_files:
    dcm_results = pd.read_csv(dcm_stats_files[0])
    print(f"\nDCM->NF results shape: {dcm_results.shape}")
    print(f"Columns: {dcm_results.columns.tolist()}")
    print(f"\nTop 20 TFs shifting DCM -> NF:")
    print(dcm_results.head(20).to_string())

# %%
# Load HCM->NF results
hcm_stats_files = sorted(STATS_OUTPUT_DIR.glob("hcm_to_nf_stats*.csv"))
print(f"HCM->NF stats files: {hcm_stats_files}")

if hcm_stats_files:
    hcm_results = pd.read_csv(hcm_stats_files[0])
    print(f"\nHCM->NF results shape: {hcm_results.shape}")
    print(f"\nTop 20 TFs shifting HCM -> NF:")
    print(hcm_results.head(20).to_string())

# %% [markdown]
# ## Step 6: Save top TFs for Phase 2D

# %%
# Save top 20 TFs for each direction (or all if fewer than 20)
TOP_N = 20

if dcm_stats_files:
    top_dcm_tfs = dcm_results.head(TOP_N)
    top_dcm_tfs.to_csv(STATS_OUTPUT_DIR / "top_tfs_dcm_to_nf.csv", index=False)
    print(f"Saved top {len(top_dcm_tfs)} DCM->NF TFs")

    # Also save as Ensembl ID list for use in Phase 2D
    if "Gene" in top_dcm_tfs.columns:
        top_dcm_ensembl = top_dcm_tfs["Gene"].tolist()
    elif "Ensembl_ID" in top_dcm_tfs.columns:
        top_dcm_ensembl = top_dcm_tfs["Ensembl_ID"].tolist()
    else:
        # Use first column as fallback
        top_dcm_ensembl = top_dcm_tfs.iloc[:, 0].tolist()
    with open(STATS_OUTPUT_DIR / "top_tfs_dcm_to_nf.pkl", "wb") as f:
        pickle.dump(top_dcm_ensembl, f)

if hcm_stats_files:
    top_hcm_tfs = hcm_results.head(TOP_N)
    top_hcm_tfs.to_csv(STATS_OUTPUT_DIR / "top_tfs_hcm_to_nf.csv", index=False)
    print(f"Saved top {len(top_hcm_tfs)} HCM->NF TFs")

    if "Gene" in top_hcm_tfs.columns:
        top_hcm_ensembl = top_hcm_tfs["Gene"].tolist()
    elif "Ensembl_ID" in top_hcm_tfs.columns:
        top_hcm_ensembl = top_hcm_tfs["Ensembl_ID"].tolist()
    else:
        top_hcm_ensembl = top_hcm_tfs.iloc[:, 0].tolist()
    with open(STATS_OUTPUT_DIR / "top_tfs_hcm_to_nf.pkl", "wb") as f:
        pickle.dump(top_hcm_ensembl, f)

# %%
print("=" * 60)
print("TREATMENT ANALYSIS COMPLETE (Phase 2C)")
print("=" * 60)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  State embeddings: {STATE_EMBS_DIR}")
print(f"  DCM->NF perturbation: {ISP_OUTPUT_DIR_DCM}")
print(f"  HCM->NF perturbation: {ISP_OUTPUT_DIR_HCM}")
print(f"  Statistics: {STATS_OUTPUT_DIR}")
