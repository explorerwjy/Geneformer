# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Summary Analysis and Publication Figures (Phase 3)
#
# Aggregates results from all phases of the cardiomyopathy reproduction study
# and generates summary tables and publication-quality figures.
#
# ## Phases summarized
# - Phase 2A: Zero-shot gene deletion (fetal cardiomyocytes)
# - Phase 2B: Disease classification (NF/HCM/DCM)
# - Phase 2C: In silico treatment analysis (TF deletion -> NF shift)
# - Phase 2D: Per-gene impact (TF x ion channel shifts)

# %%
import glob
import logging
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)

# %%
# === Paths ===
BASE_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer")
GENE_LIST_DIR = BASE_DIR / "data" / "gene_lists"

# Output directories from each phase
ZERO_SHOT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "zero_shot_deletion" / "stats"
CLASSIFICATION_DIR = BASE_DIR / "outputs" / "disease_classification"
TREATMENT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "treatment_analysis" / "stats"
GENE_IMPACT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "per_gene_impact" / "gene_stats"

SUMMARY_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Load gene name dictionaries
from geneformer import ENSEMBL_DICTIONARY_FILE

with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
    ensembl_dict = pickle.load(f)

# %% [markdown]
# ## 1. Phase 2A: Zero-shot gene deletion summary

# %%
print("=" * 60)
print("Phase 2A: Zero-shot gene deletion (fetal cardiomyocytes)")
print("=" * 60)

zero_shot_files = sorted(ZERO_SHOT_DIR.glob("*.csv")) if ZERO_SHOT_DIR.exists() else []

if zero_shot_files:
    zs_df = pd.read_csv(zero_shot_files[0])
    print(f"Total genes analyzed: {len(zs_df)}")
    print(f"\nTop 10 genes by impact:")
    print(zs_df.head(10).to_string())
    zs_df.to_csv(SUMMARY_DIR / "phase2a_zero_shot_results.csv", index=False)
else:
    print("No zero-shot results found. Run script 04 first.")
    zs_df = None

# %% [markdown]
# ## 2. Phase 2B: Disease classification summary

# %%
print("=" * 60)
print("Phase 2B: Disease classification (NF/HCM/DCM)")
print("=" * 60)

if CLASSIFICATION_DIR.exists():
    # Find the most recent classification output
    datestamp_dirs = sorted(CLASSIFICATION_DIR.glob("*"))
    datestamp_dirs = [d for d in datestamp_dirs if d.is_dir()]

    if datestamp_dirs:
        latest_dir = datestamp_dirs[-1]
        print(f"Latest classification output: {latest_dir}")

        # Look for prediction dictionaries and confusion matrices
        pred_files = list(latest_dir.glob("*pred_dict*.pkl"))
        if pred_files:
            with open(pred_files[0], "rb") as f:
                pred_dict = pickle.load(f)
            print(f"Prediction dict keys: {list(pred_dict.keys()) if isinstance(pred_dict, dict) else type(pred_dict)}")
    else:
        print("No classification output directories found.")
else:
    print("No classification directory found. Run script 05 first.")

# %% [markdown]
# ## 3. Phase 2C: Treatment analysis summary

# %%
print("=" * 60)
print("Phase 2C: In silico treatment analysis")
print("=" * 60)

dcm_stats_files = sorted(TREATMENT_DIR.glob("top_tfs_dcm_to_nf.csv")) if TREATMENT_DIR.exists() else []
hcm_stats_files = sorted(TREATMENT_DIR.glob("top_tfs_hcm_to_nf.csv")) if TREATMENT_DIR.exists() else []

dcm_top_tfs = None
hcm_top_tfs = None

if dcm_stats_files:
    dcm_top_tfs = pd.read_csv(dcm_stats_files[0])
    print(f"\nTop TFs for DCM -> NF shift ({len(dcm_top_tfs)} TFs):")
    print(dcm_top_tfs.to_string())
else:
    print("No DCM->NF treatment results found. Run script 06 first.")

if hcm_stats_files:
    hcm_top_tfs = pd.read_csv(hcm_stats_files[0])
    print(f"\nTop TFs for HCM -> NF shift ({len(hcm_top_tfs)} TFs):")
    print(hcm_top_tfs.to_string())
else:
    print("No HCM->NF treatment results found. Run script 06 first.")

# %% [markdown]
# ## 4. Phase 2D: Per-gene impact summary

# %%
print("=" * 60)
print("Phase 2D: Per-gene impact (TF x ion channel)")
print("=" * 60)

dcm_gene_file = GENE_IMPACT_DIR / "dcm_tf_ion_channel_shifts.csv" if GENE_IMPACT_DIR.exists() else None
hcm_gene_file = GENE_IMPACT_DIR / "hcm_tf_ion_channel_shifts.csv" if GENE_IMPACT_DIR.exists() else None

dcm_gene_shifts = None
hcm_gene_shifts = None

if dcm_gene_file and dcm_gene_file.exists():
    dcm_gene_shifts = pd.read_csv(dcm_gene_file)
    print(f"DCM TF x ion channel shifts: {dcm_gene_shifts.shape}")
    print(dcm_gene_shifts.head(10).to_string())
else:
    print("No DCM gene impact results found. Run script 07 first.")

if hcm_gene_file and hcm_gene_file.exists():
    hcm_gene_shifts = pd.read_csv(hcm_gene_file)
    print(f"HCM TF x ion channel shifts: {hcm_gene_shifts.shape}")
    print(hcm_gene_shifts.head(10).to_string())
else:
    print("No HCM gene impact results found. Run script 07 first.")

# %% [markdown]
# ## 5. Publication figures

# %%
# --- Figure 1: Treatment analysis - top TFs bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_alpha(0)

for ax, (tfs_df, title, color) in zip(
    axes,
    [
        (dcm_top_tfs, "DCM -> NF: Top TFs", "#E74C3C"),
        (hcm_top_tfs, "HCM -> NF: Top TFs", "#3498DB"),
    ],
):
    ax.patch.set_alpha(0)
    if tfs_df is not None and len(tfs_df) > 0:
        # Determine gene name column
        if "Gene_name" in tfs_df.columns:
            name_col = "Gene_name"
        elif "Gene" in tfs_df.columns:
            name_col = "Gene"
        else:
            name_col = tfs_df.columns[0]

        # Determine shift value column
        shift_cols = [c for c in tfs_df.columns if "shift" in c.lower() or "avg" in c.lower()]
        if shift_cols:
            val_col = shift_cols[0]
        else:
            val_col = tfs_df.columns[-1]

        display_df = tfs_df.head(15).copy()
        display_df = display_df.sort_values(val_col, ascending=True)

        ax.barh(
            range(len(display_df)),
            display_df[val_col].values,
            color=color,
            alpha=0.7,
        )
        ax.set_yticks(range(len(display_df)))
        ax.set_yticklabels(display_df[name_col].values)
        ax.set_xlabel("Goal state shift")
    ax.set_title(title)

plt.tight_layout()
fig_path = SUMMARY_DIR / "treatment_top_tfs.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", transparent=True)
plt.show()
print(f"Saved: {fig_path}")

# %%
# --- Figure 2: TF x ion channel heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_alpha(0)

for ax, (gene_df, title) in zip(
    axes,
    [
        (dcm_gene_shifts, "DCM: TF x Ion Channel Shifts"),
        (hcm_gene_shifts, "HCM: TF x Ion Channel Shifts"),
    ],
):
    ax.patch.set_alpha(0)
    if gene_df is not None and len(gene_df) > 0:
        # Try to find numeric columns for heatmap
        numeric_cols = gene_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            # Use gene names as index if available
            if "Gene_name" in gene_df.columns:
                heatmap_df = gene_df.set_index("Gene_name")[numeric_cols]
            elif "Gene" in gene_df.columns:
                heatmap_df = gene_df.set_index("Gene")[numeric_cols]
            else:
                heatmap_df = gene_df[numeric_cols]

            sns.heatmap(
                heatmap_df,
                ax=ax,
                cmap="RdBu_r",
                center=0,
                xticklabels=True,
                yticklabels=True,
            )
        else:
            ax.text(0.5, 0.5, "Insufficient data\nfor heatmap",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)

plt.tight_layout()
fig_path = SUMMARY_DIR / "tf_ion_channel_heatmap.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", transparent=True)
plt.show()
print(f"Saved: {fig_path}")

# %% [markdown]
# ## 6. Summary table

# %%
print("=" * 60)
print("CARDIOMYOPATHY REPRODUCTION STUDY - SUMMARY")
print("=" * 60)

summary_rows = []

# Phase 2A
if zs_df is not None:
    summary_rows.append({
        "Phase": "2A",
        "Analysis": "Zero-shot gene deletion",
        "Dataset": "Fetal cardiomyocytes",
        "N_genes": len(zs_df),
        "Status": "Complete",
    })
else:
    summary_rows.append({
        "Phase": "2A",
        "Analysis": "Zero-shot gene deletion",
        "Dataset": "Fetal cardiomyocytes",
        "N_genes": "N/A",
        "Status": "Not run",
    })

# Phase 2B
summary_rows.append({
    "Phase": "2B",
    "Analysis": "Disease classification (NF/HCM/DCM)",
    "Dataset": "Adult cardiomyocytes (Chaffin)",
    "N_genes": "N/A",
    "Status": "Complete" if CLASSIFICATION_DIR.exists() and datestamp_dirs else "Not run",
})

# Phase 2C
summary_rows.append({
    "Phase": "2C",
    "Analysis": "Treatment analysis (DCM->NF)",
    "Dataset": "Adult cardiomyocytes (Chaffin)",
    "N_genes": len(dcm_top_tfs) if dcm_top_tfs is not None else "N/A",
    "Status": "Complete" if dcm_top_tfs is not None else "Not run",
})
summary_rows.append({
    "Phase": "2C",
    "Analysis": "Treatment analysis (HCM->NF)",
    "Dataset": "Adult cardiomyocytes (Chaffin)",
    "N_genes": len(hcm_top_tfs) if hcm_top_tfs is not None else "N/A",
    "Status": "Complete" if hcm_top_tfs is not None else "Not run",
})

# Phase 2D
summary_rows.append({
    "Phase": "2D",
    "Analysis": "Per-gene impact (TF x ion channel)",
    "Dataset": "Adult cardiomyocytes (Chaffin)",
    "N_genes": "N/A",
    "Status": "Complete" if dcm_gene_shifts is not None else "Not run",
})

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

summary_df.to_csv(SUMMARY_DIR / "study_summary.csv", index=False)
print(f"\nSummary saved to: {SUMMARY_DIR / 'study_summary.csv'}")
