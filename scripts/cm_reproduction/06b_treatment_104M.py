# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Phase 2C: Treatment Analysis with V2-104M
#
# Uses the smaller V2-104M model for faster in silico perturbation.
# Only perturbs cardiac TFs (53 genes), not all ~13K genes.
# Each TF is perturbed individually (one ISP run per TF) to work around the
# ISP filter bug that requires ALL genes in the list to be present in each cell.
#
# Steps:
# 1. Fine-tune V2-104M classifier (NF/HCM/DCM)
# 2. Extract state embeddings (NF/HCM/DCM centroids)
# 3. Perturb each cardiac TF individually in DCM and HCM cells
# 4. Compute goal_state_shift stats to rank TFs
# 5. Save top TFs for Phase 2D (per-gene ion channel impact)

# %%
import datetime
import glob
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geneformer import Classifier, EmbExtractor, InSilicoPerturber, InSilicoPerturberStats

logging.basicConfig(level=logging.INFO)

# %%
# === Paths ===
BASE_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer")
MODEL_DIR = BASE_DIR / "models" / "Geneformer" / "Geneformer-V2-104M"
INPUT_DATA = BASE_DIR / "data" / "tokenized" / "chaffin_cardiomyocytes.dataset"
GENE_LIST_DIR = BASE_DIR / "data" / "gene_lists"

OUTPUT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "treatment_104M"
CLASSIFIER_DIR = OUTPUT_DIR / "classifier"
STATE_EMBS_DIR = OUTPUT_DIR / "state_embs"
ISP_OUTPUT_DIR_DCM = OUTPUT_DIR / "isp_dcm_to_nf"
ISP_OUTPUT_DIR_HCM = OUTPUT_DIR / "isp_hcm_to_nf"
STATS_DIR = OUTPUT_DIR / "stats"

for d in [CLASSIFIER_DIR, STATE_EMBS_DIR, ISP_OUTPUT_DIR_DCM, ISP_OUTPUT_DIR_HCM, STATS_DIR]:
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
# === Configuration ===
MAX_NCELLS = 500  # smaller for speed
FORWARD_BATCH_SIZE = 25  # CellClassifier uses more VRAM than expected
EMB_BATCH_SIZE = 50

# Patient-level split (same as 316M training)
dcm_donors = ["P1290", "P1300", "P1304", "P1358", "P1371", "P1430", "P1437",
               "P1472", "P1504", "P1606", "P1617"]
hcm_donors = ["P1422", "P1425", "P1447", "P1462", "P1479", "P1508", "P1510",
               "P1602", "P1630", "P1631", "P1685", "P1707", "P1722", "P1726",
               "P1735"]
nf_donors = ["P1515", "P1516", "P1539", "P1540", "P1547", "P1549", "P1558",
              "P1561", "P1582", "P1600", "P1603", "P1610", "P1622", "P1678",
              "P1702", "P1718"]

train_ids = (["P1290", "P1300", "P1304", "P1358", "P1371", "P1430", "P1472", "P1504"]
             + ["P1422", "P1425", "P1447", "P1462", "P1479", "P1508", "P1602",
                "P1630", "P1631", "P1707", "P1722"]
             + ["P1515", "P1539", "P1540", "P1547", "P1549", "P1558", "P1561",
                "P1582", "P1600", "P1603", "P1610"])
eval_ids = ["P1606", "P1510", "P1726", "P1622", "P1678"]
test_ids = ["P1437", "P1617", "P1685", "P1735", "P1516", "P1702", "P1718"]

# %% [markdown]
# ## Step 1: Fine-tune V2-104M classifier

# %%
print("=" * 60)
print("STEP 1: Fine-tuning V2-104M classifier")
print("=" * 60)

output_prefix = "disease_104M"

# Check if classifier already trained
model_candidates = sorted(glob.glob(str(CLASSIFIER_DIR / "*cellClassifier*" / "ksplit1")))

if model_candidates:
    FINETUNED_MODEL = model_candidates[-1]
    print(f"Classifier already trained, skipping. Using: {FINETUNED_MODEL}")
else:
    training_args = {
        "num_train_epochs": 1,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 12,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "seed": 42,
    }

    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "disease", "states": "all"},
        filter_data=None,
        training_args=training_args,
        freeze_layers=4,  # V2-104M has 6 layers; freeze bottom 4
        num_crossval_splits=1,
        forward_batch_size=FORWARD_BATCH_SIZE,
        model_version="V2",
        nproc=1,
        ngpu=1,
    )

    # Prepare data
    train_test_split = {
        "attr_key": "donor_id",
        "train": train_ids + eval_ids,
        "test": test_ids,
    }

    cc.prepare_data(
        input_data_file=str(INPUT_DATA),
        output_directory=str(CLASSIFIER_DIR),
        output_prefix=output_prefix,
        split_id_dict=train_test_split,
    )

    # Train
    train_valid_split = {
        "attr_key": "donor_id",
        "train": train_ids,
        "eval": eval_ids,
    }

    all_metrics = cc.validate(
        model_directory=str(MODEL_DIR),
        prepared_input_data_file=f"{CLASSIFIER_DIR}/{output_prefix}_labeled_train.dataset",
        id_class_dict_file=f"{CLASSIFIER_DIR}/{output_prefix}_id_class_dict.pkl",
        output_directory=str(CLASSIFIER_DIR),
        output_prefix=output_prefix,
        split_id_dict=train_valid_split,
    )

    print(f"Validation Macro F1: {all_metrics['macro_f1'][0]:.4f}")
    print(f"Validation Accuracy: {all_metrics['acc'][0]:.4f}")

    model_candidates = sorted(glob.glob(str(CLASSIFIER_DIR / "*cellClassifier*" / "ksplit1")))
    if not model_candidates:
        print("ERROR: No fine-tuned model found!")
        sys.exit(1)

    FINETUNED_MODEL = model_candidates[-1]
    print(f"Fine-tuned model: {FINETUNED_MODEL}")

# %% [markdown]
# ## Step 2: Extract state embeddings

# %%
print("=" * 60)
print("STEP 2: Extracting state embeddings")
print("=" * 60)

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

state_embs_file = STATE_EMBS_DIR / "disease_state_embs.pkl"
if state_embs_file.exists():
    print(f"Loading cached state embeddings from {state_embs_file}")
    with open(state_embs_file, "rb") as f:
        state_embs_dict = pickle.load(f)
else:
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
        nproc=1,
    )

    state_embs_dict = embex.get_state_embs(
        cell_states_to_model=dcm_states,
        model_directory=FINETUNED_MODEL,
        input_data_file=str(INPUT_DATA),
        output_directory=str(STATE_EMBS_DIR),
        output_prefix="disease_state_embs",
    )

print(f"State embeddings: {list(state_embs_dict.keys())}")
for state, emb in state_embs_dict.items():
    print(f"  {state}: shape {emb.shape}")

# %% [markdown]
# ## Step 3: In silico perturbation — DCM to NF (TFs only)
#
# Loop over each cardiac TF individually to avoid the ISP filter bug
# (filter_data_by_tokens requires ALL genes in the list to be present in each cell).
# With a single-element list, the filter passes for any cell containing that TF.

# %%
print("=" * 60)
print(f"ISP: DCM -> NF (perturbing {len(cardiac_tfs)} TFs individually)")
print("=" * 60)

for i, tf in enumerate(cardiac_tfs):
    # Skip if output already exists
    existing = list(ISP_OUTPUT_DIR_DCM.glob(f"in_silico_delete_dcm_tf_{i:03d}_*_raw.pickle"))
    if existing:
        print(f"--- TF {i+1}/{len(cardiac_tfs)}: {tf} --- SKIPPED (output exists)")
        continue

    print(f"\n--- TF {i+1}/{len(cardiac_tfs)}: {tf} ---")

    isp_dcm = InSilicoPerturber(
        perturb_type="delete",
        genes_to_perturb=[tf],
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
        nproc=1,
    )

    isp_dcm.perturb_data(
        model_directory=FINETUNED_MODEL,
        input_data_file=str(INPUT_DATA),
        output_directory=str(ISP_OUTPUT_DIR_DCM),
        output_prefix=f"dcm_tf_{i:03d}",
    )

# %% [markdown]
# ## Step 4: In silico perturbation — HCM to NF (TFs only)

# %%
print("=" * 60)
print(f"ISP: HCM -> NF (perturbing {len(cardiac_tfs)} TFs individually)")
print("=" * 60)

for i, tf in enumerate(cardiac_tfs):
    # Skip if output already exists
    existing = list(ISP_OUTPUT_DIR_HCM.glob(f"in_silico_delete_hcm_tf_{i:03d}_*_raw.pickle"))
    if existing:
        print(f"--- TF {i+1}/{len(cardiac_tfs)}: {tf} --- SKIPPED (output exists)")
        continue

    print(f"\n--- TF {i+1}/{len(cardiac_tfs)}: {tf} ---")

    isp_hcm = InSilicoPerturber(
        perturb_type="delete",
        genes_to_perturb=[tf],
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
        nproc=1,
    )

    isp_hcm.perturb_data(
        model_directory=FINETUNED_MODEL,
        input_data_file=str(INPUT_DATA),
        output_directory=str(ISP_OUTPUT_DIR_HCM),
        output_prefix=f"hcm_tf_{i:03d}",
    )

# %% [markdown]
# ## Step 5: Compute stats (goal_state_shift)
#
# All 53 per-TF pickle files are in a single flat directory, so ISPStats can
# read them all. We use `genes_perturbed="all"` because each file contains
# a different gene's perturbation result and ISPStats aggregates across files.

# %%
print("=" * 60)
print("Computing goal_state_shift stats")
print("=" * 60)

# DCM -> NF stats
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
    output_directory=str(STATS_DIR),
    output_prefix="dcm_to_nf",
)

# HCM -> NF stats
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
    output_directory=str(STATS_DIR),
    output_prefix="hcm_to_nf",
)

# %% [markdown]
# ## Step 6: Analyze and rank TFs by goal state shift

# %%
print("=" * 60)
print("Analyzing results")
print("=" * 60)

# Load stats
dcm_stats_files = sorted(STATS_DIR.glob("dcm_to_nf*.csv"))
hcm_stats_files = sorted(STATS_DIR.glob("hcm_to_nf*.csv"))

print(f"DCM stats files: {dcm_stats_files}")
print(f"HCM stats files: {hcm_stats_files}")

dcm_df = pd.read_csv(dcm_stats_files[0])
hcm_df = pd.read_csv(hcm_stats_files[0])

print(f"DCM results: {dcm_df.shape}")
print(f"HCM results: {hcm_df.shape}")

# %%
# Identify the shift column (varies by ISP stats version)
shift_col = None
for col in dcm_df.columns:
    if "shift" in col.lower() or "goal" in col.lower():
        shift_col = col
        break
if shift_col is None:
    numeric_cols = dcm_df.select_dtypes(include=[np.number]).columns.tolist()
    shift_col = numeric_cols[0] if numeric_cols else None

print(f"Using shift column: {shift_col}")
print(f"Columns: {dcm_df.columns.tolist()}")

# %%
# All results are cardiac TFs since we only perturbed TFs
print("\n=== Cardiac TFs ranked by DCM -> NF shift ===")
dcm_sorted = dcm_df.sort_values(shift_col, ascending=False)
print(dcm_sorted.to_string())

print("\n=== Cardiac TFs ranked by HCM -> NF shift ===")
hcm_sorted = hcm_df.sort_values(shift_col, ascending=False)
print(hcm_sorted.to_string())

# %%
# Save top TFs for Phase 2D
top_n = 20
gene_col = "Gene" if "Gene" in dcm_sorted.columns else "Ensembl_ID"

dcm_top = dcm_sorted.head(top_n)
hcm_top = hcm_sorted.head(top_n)

all_top_tfs = list(set(dcm_top[gene_col].tolist() + hcm_top[gene_col].tolist()))
print(f"\nUnique top TFs (union of both directions): {len(all_top_tfs)}")

with open(STATS_DIR / "top_tfs_dcm_to_nf.pkl", "wb") as f:
    pickle.dump(dcm_top[gene_col].tolist(), f)
with open(STATS_DIR / "top_tfs_hcm_to_nf.pkl", "wb") as f:
    pickle.dump(hcm_top[gene_col].tolist(), f)
with open(STATS_DIR / "top_tfs_combined.pkl", "wb") as f:
    pickle.dump(all_top_tfs, f)

# %%
# Save full results
dcm_sorted.to_csv(STATS_DIR / "dcm_to_nf_cardiac_tfs.csv", index=False)
hcm_sorted.to_csv(STATS_DIR / "hcm_to_nf_cardiac_tfs.csv", index=False)

# %% [markdown]
# ## Step 7: Visualization

# %%
if len(dcm_sorted) > 0 and len(hcm_sorted) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_alpha(0)

    name_col = "Gene_name" if "Gene_name" in dcm_sorted.columns else gene_col

    # DCM -> NF: top cardiac TFs
    ax = axes[0]
    ax.patch.set_alpha(0)
    top_dcm = dcm_sorted.head(15)
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in top_dcm[shift_col]]
    ax.barh(range(len(top_dcm)), top_dcm[shift_col].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_dcm)))
    ax.set_yticklabels(top_dcm[name_col].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Shift toward NF")
    ax.set_title("DCM -> NF: Top Cardiac TFs")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # HCM -> NF: top cardiac TFs
    ax = axes[1]
    ax.patch.set_alpha(0)
    top_hcm = hcm_sorted.head(15)
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in top_hcm[shift_col]]
    ax.barh(range(len(top_hcm)), top_hcm[shift_col].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_hcm)))
    ax.set_yticklabels(top_hcm[name_col].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Shift toward NF")
    ax.set_title("HCM -> NF: Top Cardiac TFs")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(STATS_DIR / "treatment_cardiac_tfs.png", dpi=150, transparent=True, bbox_inches="tight")
    print(f"Figure saved to {STATS_DIR / 'treatment_cardiac_tfs.png'}")

# %%
print("\n" + "=" * 60)
print("TREATMENT ANALYSIS COMPLETE (V2-104M)")
print("=" * 60)
print(f"Results saved to: {STATS_DIR}")
print(f"TFs analyzed: DCM={len(dcm_df)}, HCM={len(hcm_df)}")
