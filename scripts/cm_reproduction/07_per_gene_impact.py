# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Phase 2D: Per-Gene Ion Channel Impact Extraction
#
# For top TFs from Phase 2C, re-run perturbation with emb_mode="cell_and_gene"
# to get per-gene embedding shifts. Filter for ion channel genes to prepare
# for openCARP integration.
#
# Output: TF x ion_channel matrix of embedding shifts, showing how each
# therapeutic TF affects each ion channel gene's representation.

# %%
import glob
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geneformer import EmbExtractor, InSilicoPerturber
from geneformer import ENSEMBL_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE

logging.basicConfig(level=logging.INFO)

# %%
# === Paths ===
BASE_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer")
MODEL_DIR = BASE_DIR / "models" / "Geneformer" / "Geneformer-V2-104M"
INPUT_DATA = BASE_DIR / "data" / "tokenized" / "chaffin_cardiomyocytes.dataset"
GENE_LIST_DIR = BASE_DIR / "data" / "gene_lists"

# Phase 2C outputs (reuse classifier and state embeddings)
PHASE2C_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "treatment_104M"
CLASSIFIER_DIR = PHASE2C_DIR / "classifier"
STATE_EMBS_DIR = PHASE2C_DIR / "state_embs"

# Phase 2D output
OUTPUT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "per_gene_impact"
ISP_OUTPUT_DIR = OUTPUT_DIR / "isp_gene_level"
STATS_DIR = OUTPUT_DIR / "stats"
for d in [OUTPUT_DIR, ISP_OUTPUT_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Batch sizes (cell_and_gene mode uses more VRAM)
FORWARD_BATCH_SIZE = 20
MAX_NCELLS = 500

# %%
# === Load gene dictionaries ===
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dict = pickle.load(f)
token_to_ens = {v: k for k, v in token_dict.items()}

with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
    ens_dict = pickle.load(f)
ens_to_name = {v: k for k, v in ens_dict.items()}

# %%
# === Load top TFs from Phase 2C ===
with open(PHASE2C_DIR / "stats" / "top_tfs_combined.pkl", "rb") as f:
    top_tf_tokens = pickle.load(f)
print(f"Top TFs from Phase 2C: {len(top_tf_tokens)}")
for t in top_tf_tokens:
    ens = token_to_ens.get(t, "?")
    name = ens_to_name.get(ens, "?")
    print(f"  {t} -> {name} ({ens})")

# === Load ion channel genes ===
with open(GENE_LIST_DIR / "ion_channel_genes.pkl", "rb") as f:
    ion_channel_ensembls = pickle.load(f)
ion_channel_tokens = set()
for ens in ion_channel_ensembls:
    if ens in token_dict:
        ion_channel_tokens.add(token_dict[ens])
print(f"\nIon channel genes: {len(ion_channel_ensembls)} "
      f"({len(ion_channel_tokens)} with tokens)")
for ens in ion_channel_ensembls:
    name = ens_to_name.get(ens, "?")
    tid = token_dict.get(ens, "?")
    print(f"  {name} ({ens}) -> token {tid}")

# %%
# === Load fine-tuned classifier (from Phase 2C) ===
model_candidates = sorted(
    glob.glob(str(CLASSIFIER_DIR / "*cellClassifier*" / "ksplit1"))
)
if not model_candidates:
    raise FileNotFoundError("No fine-tuned classifier found from Phase 2C")
FINETUNED_MODEL = model_candidates[-1]
print(f"\nUsing classifier: {FINETUNED_MODEL}")

# === Load state embeddings (from Phase 2C) ===
state_embs_file = STATE_EMBS_DIR / "disease_state_embs.pkl"
if not state_embs_file.exists():
    raise FileNotFoundError("No state embeddings found from Phase 2C")
with open(state_embs_file, "rb") as f:
    state_embs_dict = pickle.load(f)
print(f"State embeddings: {list(state_embs_dict.keys())}")

# %%
# === Cell state definitions ===
dcm_states = {
    "state_key": "disease",
    "start_state": "DCM",
    "goal_state": "NF",
    "alt_states": ["HCM"],
}

# %% [markdown]
# ## Per-gene perturbation (cell_and_gene mode)
#
# For each top TF, delete it in DCM cells and capture per-gene embedding shifts.
# This tells us how each gene's learned representation changes when the TF is removed.
# We then extract ion channel genes to understand TF -> ion channel relationships.

# %%
print("=" * 60)
print(f"ISP: Per-gene impact for {len(top_tf_tokens)} top TFs")
print(f"Mode: cls_and_gene (gene-level embedding shifts)")
print("=" * 60)

for i, tf_token in enumerate(top_tf_tokens):
    tf_ens = token_to_ens.get(tf_token, "?")
    tf_name = ens_to_name.get(tf_ens, "?")

    # Skip if output already exists
    existing = list(ISP_OUTPUT_DIR.glob(
        f"in_silico_delete_tf_{i:03d}_*gene_embs_dict*_raw.pickle"
    ))
    if existing:
        print(f"--- TF {i+1}/{len(top_tf_tokens)}: {tf_name} --- SKIPPED (output exists)")
        continue

    print(f"\n--- TF {i+1}/{len(top_tf_tokens)}: {tf_name} ({tf_ens}) ---")

    isp = InSilicoPerturber(
        perturb_type="delete",
        genes_to_perturb=[tf_ens],
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
        nproc=1,
    )

    isp.perturb_data(
        model_directory=FINETUNED_MODEL,
        input_data_file=str(INPUT_DATA),
        output_directory=str(ISP_OUTPUT_DIR),
        output_prefix=f"tf_{i:03d}",
    )

print("\nPer-gene ISP complete!")

# %% [markdown]
# ## Extract ion channel shifts from gene-level output

# %%
print("=" * 60)
print("Extracting ion channel gene shifts from gene_embs_dict files")
print("=" * 60)

gene_embs_files = sorted(ISP_OUTPUT_DIR.glob("*gene_embs_dict*_raw.pickle"))
print(f"Found {len(gene_embs_files)} gene_embs_dict files")

results = []
for gf in gene_embs_files:
    with open(gf, "rb") as f:
        gene_embs_dict = pickle.load(f)

    # gene_embs_dict keys: (perturbed_token, affected_gene_token)
    # values: list of cosine similarities (one per cell)
    for (perturbed_token, affected_token), cos_sims in gene_embs_dict.items():
        # Handle both tuple and single token for perturbed
        if isinstance(perturbed_token, tuple):
            perturbed_token = perturbed_token[0]

        perturbed_ens = token_to_ens.get(perturbed_token, str(perturbed_token))
        perturbed_name = ens_to_name.get(perturbed_ens, perturbed_ens)

        affected_ens = token_to_ens.get(affected_token, str(affected_token))
        affected_name = ens_to_name.get(affected_ens, affected_ens)

        if isinstance(cos_sims, list):
            mean_shift = np.mean(cos_sims)
            n_cells = len(cos_sims)
        else:
            mean_shift = cos_sims
            n_cells = 1

        # Check if affected gene is an ion channel
        is_ion_channel = affected_ens in ion_channel_ensembls

        results.append({
            "tf_name": perturbed_name,
            "tf_ensembl": perturbed_ens,
            "tf_token": perturbed_token,
            "affected_gene": affected_name,
            "affected_ensembl": affected_ens,
            "affected_token": affected_token,
            "mean_cosine_shift": mean_shift,
            "n_cells": n_cells,
            "is_ion_channel": is_ion_channel,
        })

df_all = pd.DataFrame(results)
print(f"Total TF x gene entries: {len(df_all)}")
print(f"Unique TFs: {df_all['tf_name'].nunique()}")
print(f"Unique affected genes: {df_all['affected_gene'].nunique()}")

# Save full results
df_all.to_csv(STATS_DIR / "all_gene_shifts.csv", index=False)

# %%
# === Filter for ion channel genes ===
df_ic = df_all[df_all["is_ion_channel"]].copy()
df_ic = df_ic.sort_values("mean_cosine_shift", ascending=True)

print(f"\n{'='*60}")
print(f"Ion channel gene shifts: {len(df_ic)} entries")
print(f"TFs with ion channel effects: {df_ic['tf_name'].nunique()}")
print(f"Ion channels found: {df_ic['affected_gene'].nunique()}")
print(f"{'='*60}")

if len(df_ic) > 0:
    # Save ion channel results
    df_ic.to_csv(STATS_DIR / "tf_ion_channel_shifts.csv", index=False)

    # Create pivot table: TF x ion channel
    pivot = df_ic.pivot_table(
        index="tf_name",
        columns="affected_gene",
        values="mean_cosine_shift",
        aggfunc="mean",
    )

    # Sort by mean absolute shift
    pivot["mean_abs_shift"] = pivot.abs().mean(axis=1)
    pivot = pivot.sort_values("mean_abs_shift", ascending=False)
    pivot = pivot.drop(columns="mean_abs_shift")

    pivot.to_csv(STATS_DIR / "tf_ion_channel_pivot.csv")
    print("\nTF x Ion Channel pivot table:")
    print(pivot.to_string())

    # === Heatmap ===
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8),
                                     max(6, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-np.max(np.abs(pivot.values)),
                   vmax=np.max(np.abs(pivot.values)))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Ion Channel Gene")
    ax.set_ylabel("Deleted TF")
    ax.set_title("Embedding shift per ion channel when TF is deleted\n"
                 "(negative = larger change, potential disruption)")
    plt.colorbar(im, ax=ax, label="Mean cosine similarity shift")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.tight_layout()
    fig.savefig(STATS_DIR / "tf_ion_channel_heatmap.png", dpi=150, transparent=True)
    print(f"\nHeatmap saved to {STATS_DIR / 'tf_ion_channel_heatmap.png'}")

    # === Top interactions ===
    print(f"\nTop 20 TF -> ion channel interactions (largest shifts):")
    print(df_ic.head(20)[["tf_name", "affected_gene", "mean_cosine_shift", "n_cells"]]
          .to_string(index=False))
else:
    print("No ion channel genes found in perturbation output.")

# %%
print(f"\n{'='*60}")
print("PHASE 2D COMPLETE")
print(f"{'='*60}")
print(f"Results saved to: {STATS_DIR}")
print(f"  - all_gene_shifts.csv: All TF x gene shifts")
print(f"  - tf_ion_channel_shifts.csv: Filtered for ion channels")
print(f"  - tf_ion_channel_pivot.csv: Pivot table (TF rows x IC columns)")
print(f"  - tf_ion_channel_heatmap.png: Visualization")
