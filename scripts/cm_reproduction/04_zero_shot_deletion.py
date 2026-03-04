# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Zero-shot in silico deletion of genes in fetal cardiomyocytes
#
# Uses the pretrained Geneformer V2-316M model to delete genes individually
# in fetal cardiomyocytes and identify which disease-associated genes have the
# greatest impact on cell embeddings using a Gaussian mixture model.
#
# ## Approach
# 1. Run InSilicoPerturber with `genes_to_perturb="all"` to delete each detected
#    gene individually and measure the cosine shift in CLS embeddings.
# 2. Use InSilicoPerturberStats with `mode="mixture_model"` to fit a two-component
#    GMM (impact vs no-impact) and classify each gene.
# 3. Compare disease genes vs control genes to check whether cardiomyopathy genes
#    are enriched in the impact component.

# %%
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from geneformer import InSilicoPerturber, InSilicoPerturberStats

logging.basicConfig(level=logging.INFO)

# %%
# === Paths ===
BASE_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer")
MODEL_DIR = BASE_DIR / "models" / "Geneformer" / "Geneformer-V2-316M"
INPUT_DATA = BASE_DIR / "data" / "tokenized" / "fetal_cardiomyocytes.dataset"
GENE_LIST_DIR = BASE_DIR / "data" / "gene_lists"

OUTPUT_DIR = BASE_DIR / "outputs" / "cm_reproduction" / "zero_shot_deletion"
ISP_OUTPUT_DIR = OUTPUT_DIR / "isp_output"
STATS_OUTPUT_DIR = OUTPUT_DIR / "stats"

for d in [ISP_OUTPUT_DIR, STATS_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# %%
# === Load gene lists ===
with open(GENE_LIST_DIR / "disease_combined.pkl", "rb") as f:
    disease_genes = pickle.load(f)

with open(GENE_LIST_DIR / "hyperlipidaemia_control.pkl", "rb") as f:
    control_genes = pickle.load(f)

print(f"Disease genes: {len(disease_genes)}")
print(f"Control genes: {len(control_genes)}")

# %%
# === Configuration ===
# Set MAX_NCELLS to a small number for smoke testing, or 2000 for full run
MAX_NCELLS = 2000  # Set to 50 for quick test, 2000 for full run
FORWARD_BATCH_SIZE = 200

# %% [markdown]
# ## Step 1: Run in silico perturbation (delete each gene individually)

# %%
isp = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb="all",
    combos=0,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    filter_data={"cell_type": ["cardiac muscle cell"]},
    max_ncells=MAX_NCELLS,
    emb_layer=-1,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version="V2",
    nproc=10,
)

isp.perturb_data(
    model_directory=str(MODEL_DIR),
    input_data_file=str(INPUT_DATA),
    output_directory=str(ISP_OUTPUT_DIR),
    output_prefix="fetal_cm",
)

# %% [markdown]
# ## Step 2: Compute mixture model stats

# %%
ispstats = InSilicoPerturberStats(
    mode="mixture_model",
    genes_perturbed="all",
    combos=0,
    model_version="V2",
)

ispstats.get_stats(
    input_data_directory=str(ISP_OUTPUT_DIR),
    null_dist_data_directory=None,
    output_directory=str(STATS_OUTPUT_DIR),
    output_prefix="fetal_cm_mixture",
)

# %% [markdown]
# ## Step 3: Load results and compare disease vs control genes

# %%
# Find the stats CSV output
stats_files = list(STATS_OUTPUT_DIR.glob("*.csv"))
print(f"Stats output files: {stats_files}")

stats_df = pd.read_csv(stats_files[0])
print(f"\nStats shape: {stats_df.shape}")
print(f"Columns: {stats_df.columns.tolist()}")
print(f"\nTop 20 genes by impact:")
print(stats_df.head(20).to_string())

# %%
# Annotate genes as disease, control, or other
disease_set = set(disease_genes)
control_set = set(control_genes)

stats_df["gene_category"] = stats_df["Ensembl_ID"].apply(
    lambda x: "disease"
    if x in disease_set
    else ("control" if x in control_set else "other")
)

print("\nGene category counts in results:")
print(stats_df["gene_category"].value_counts())

# %%
# Compare disease vs control genes
disease_df = stats_df[stats_df["gene_category"] == "disease"].copy()
control_df = stats_df[stats_df["gene_category"] == "control"].copy()

print(f"\nDisease genes found in results: {len(disease_df)} / {len(disease_genes)}")
print(f"Control genes found in results: {len(control_df)} / {len(control_genes)}")

if len(disease_df) > 0:
    print(f"\n--- Disease genes ---")
    print(
        disease_df[["Gene_name", "Ensembl_ID", "Test_avg_shift",
                     "Impact_component", "Impact_component_percent"]]
        .sort_values("Test_avg_shift", ascending=False)
        .to_string(index=False)
    )

if len(control_df) > 0:
    print(f"\n--- Control genes ---")
    print(
        control_df[["Gene_name", "Ensembl_ID", "Test_avg_shift",
                     "Impact_component", "Impact_component_percent"]]
        .sort_values("Test_avg_shift", ascending=False)
        .to_string(index=False)
    )

# %%
# Statistical comparison
if len(disease_df) > 0 and len(control_df) > 0:
    # Mann-Whitney U test on average shifts
    stat, pval = mannwhitneyu(
        disease_df["Test_avg_shift"].dropna(),
        control_df["Test_avg_shift"].dropna(),
        alternative="greater",
    )
    print(f"\nMann-Whitney U test (disease > control shift):")
    print(f"  U statistic: {stat:.2f}")
    print(f"  p-value: {pval:.4e}")

    # Fisher's exact test on impact component membership
    disease_impact = (disease_df["Impact_component"] == 1).sum()
    disease_no_impact = (disease_df["Impact_component"] == 0).sum()
    control_impact = (control_df["Impact_component"] == 1).sum()
    control_no_impact = (control_df["Impact_component"] == 0).sum()

    table = [[disease_impact, disease_no_impact],
             [control_impact, control_no_impact]]
    odds_ratio, fisher_pval = fisher_exact(table, alternative="greater")
    print(f"\nFisher's exact test (disease enriched in impact component):")
    print(f"  Disease: {disease_impact}/{len(disease_df)} in impact component")
    print(f"  Control: {control_impact}/{len(control_df)} in impact component")
    print(f"  Odds ratio: {odds_ratio:.2f}")
    print(f"  p-value: {fisher_pval:.4e}")

# %% [markdown]
# ## Step 4: Visualization

# %%
if len(disease_df) > 0 and len(control_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_alpha(0)

    # Plot 1: Box plot of average shifts
    ax = axes[0]
    ax.patch.set_alpha(0)
    data_to_plot = [
        disease_df["Test_avg_shift"].dropna().values,
        control_df["Test_avg_shift"].dropna().values,
    ]
    bp = ax.boxplot(
        data_to_plot,
        labels=["Disease genes", "Control genes"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor("#E74C3C")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#3498DB")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_ylabel("Average cosine shift (CLS embedding)")
    ax.set_title("Embedding shift upon gene deletion")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Plot 2: Impact component fraction
    ax = axes[1]
    ax.patch.set_alpha(0)
    categories = ["Disease", "Control"]
    impact_fracs = [
        disease_impact / len(disease_df) if len(disease_df) > 0 else 0,
        control_impact / len(control_df) if len(control_df) > 0 else 0,
    ]
    bars = ax.bar(categories, impact_fracs, color=["#E74C3C", "#3498DB"], alpha=0.7)
    ax.set_ylabel("Fraction in impact component")
    ax.set_title("GMM impact component membership")
    ax.set_ylim(0, 1)
    for bar, frac in zip(bars, impact_fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{frac:.1%}",
            ha="center",
            fontsize=11,
        )

    # Plot 3: Distribution of shifts
    ax = axes[2]
    ax.patch.set_alpha(0)
    all_shifts = stats_df["Test_avg_shift"].dropna()
    ax.hist(all_shifts, bins=50, alpha=0.4, color="gray", label="All genes")
    ax.hist(
        disease_df["Test_avg_shift"].dropna(),
        bins=20,
        alpha=0.7,
        color="#E74C3C",
        label="Disease genes",
    )
    ax.hist(
        control_df["Test_avg_shift"].dropna(),
        bins=10,
        alpha=0.7,
        color="#3498DB",
        label="Control genes",
    )
    ax.set_xlabel("Average cosine shift")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of gene deletion effects")
    ax.legend()

    plt.tight_layout()
    fig_path = STATS_OUTPUT_DIR / "disease_vs_control_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Figure saved to {fig_path}")

# %%
# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total genes analyzed: {len(stats_df)}")
if len(disease_df) > 0:
    print(
        f"Disease genes - mean shift: {disease_df['Test_avg_shift'].mean():.6f}, "
        f"impact fraction: {disease_impact}/{len(disease_df)}"
    )
if len(control_df) > 0:
    print(
        f"Control genes - mean shift: {control_df['Test_avg_shift'].mean():.6f}, "
        f"impact fraction: {control_impact}/{len(control_df)}"
    )
