"""
Step 5b: Analyze in silico perturbation results from raw pickle files.

Directly loads perturbation cosine shifts and computes statistics
for our curated liver TF/signaling gene list, bypassing the slow
built-in mixture model.
"""

import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

# ── Paths ───────────────────────────────────────────────────────────
ISP_OUTPUT_DIR = "/home/jw3514/Work/Geneformer/outputs/liver_perturbation/260311092219/isp_raw"
OUTPUT_DIR = "/home/jw3514/Work/Geneformer/outputs/liver_perturbation/260311092219"

# ── Load token dictionary for gene name lookup ───────────────────────
with open("/home/jw3514/Work/Geneformer/Geneformer/geneformer/token_dictionary_gc104M.pkl", "rb") as f:
    token_dict = pickle.load(f)
token_to_ensembl = {v: k for k, v in token_dict.items()}

with open("/home/jw3514/Work/Geneformer/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl", "rb") as f:
    gene_name_id = pickle.load(f)
ensembl_to_name = {v: k for k, v in gene_name_id.items()}

# ── Curated liver genes (Ensembl -> name) ────────────────────────────
LIVER_GENES = {
    "ENSG00000101076": "HNF4A", "ENSG00000135100": "HNF1A",
    "ENSG00000275410": "HNF1B", "ENSG00000129514": "FOXA1",
    "ENSG00000125798": "FOXA2", "ENSG00000170608": "FOXA3",
    "ENSG00000187079": "TEAD1", "ENSG00000074219": "TEAD2",
    "ENSG00000007866": "TEAD3", "ENSG00000197905": "TEAD4",
    "ENSG00000137693": "YAP1", "ENSG00000018408": "WWTR1",
    "ENSG00000133703": "KRAS", "ENSG00000174775": "HRAS",
    "ENSG00000213281": "NRAS", "ENSG00000157764": "BRAF",
    "ENSG00000132155": "RAF1",
    "ENSG00000186951": "PPARA", "ENSG00000132170": "PPARG",
    "ENSG00000109819": "PPARGC1A", "ENSG00000072310": "SREBF1",
    "ENSG00000198911": "SREBF2", "ENSG00000012504": "NR1H4",
    "ENSG00000025434": "NR1H3",
    "ENSG00000245848": "CEBPA", "ENSG00000172216": "CEBPB",
    "ENSG00000135111": "TBX3", "ENSG00000136574": "GATA4",
    "ENSG00000141448": "GATA6", "ENSG00000125398": "SOX9",
    "ENSG00000117707": "PROX1",
    "ENSG00000105329": "TGFB1", "ENSG00000175387": "SMAD2",
    "ENSG00000166949": "SMAD3", "ENSG00000141646": "SMAD4",
    "ENSG00000124216": "SNAI1", "ENSG00000019549": "SNAI2",
    "ENSG00000122691": "TWIST1",
    "ENSG00000173039": "RELA", "ENSG00000109320": "NFKB1",
    "ENSG00000077150": "NFKB2", "ENSG00000168610": "STAT3",
    "ENSG00000177606": "JUN", "ENSG00000170345": "FOS",
    "ENSG00000232810": "TNF", "ENSG00000136244": "IL6",
    "ENSG00000148400": "NOTCH1", "ENSG00000134250": "NOTCH2",
    "ENSG00000114315": "HES1", "ENSG00000101384": "JAG1",
    "ENSG00000116044": "NFE2L2", "ENSG00000100644": "HIF1A",
    "ENSG00000141510": "TP53",
    "ENSG00000168036": "CTNNB1", "ENSG00000134982": "APC",
    "ENSG00000168646": "AXIN2",
    "ENSG00000107796": "ACTA2", "ENSG00000108821": "COL1A1",
    "ENSG00000168542": "COL3A1", "ENSG00000113721": "PDGFRB",
    "ENSG00000163631": "ALB", "ENSG00000081051": "AFP",
    "ENSG00000084674": "APOB", "ENSG00000160868": "CYP3A4",
    "ENSG00000140505": "CYP1A2",
}

# Token IDs for our liver genes
liver_token_ids = {}
for ens_id, gene_name in LIVER_GENES.items():
    if ens_id in token_dict:
        liver_token_ids[token_dict[ens_id]] = gene_name

print(f"Liver genes with token IDs: {len(liver_token_ids)}")

# ── Load all raw pickle files ────────────────────────────────────────
print("\nLoading raw perturbation results...")
pickle_files = sorted(glob.glob(f"{ISP_OUTPUT_DIR}/*_raw.pickle"))
print(f"Found {len(pickle_files)} pickle files")

# Aggregate cosine shifts per gene (token_id -> list of shifts)
gene_shifts = defaultdict(list)
all_gene_shifts = defaultdict(list)  # for global stats

for pf in pickle_files:
    with open(pf, "rb") as f:
        batch_dict = pickle.load(f)

    for key, values in batch_dict.items():
        if isinstance(key, tuple) and len(key) == 2 and key[1] == "cell_emb":
            token_id = key[0]
            all_gene_shifts[token_id].extend(values)
            if token_id in liver_token_ids:
                gene_shifts[token_id].extend(values)

print(f"Total genes with perturbation data: {len(all_gene_shifts)}")
print(f"Liver genes with perturbation data: {len(gene_shifts)}/{len(liver_token_ids)}")

# ── Compute statistics for all genes ─────────────────────────────────
# Raw values are cosine similarities (high = unchanged, low = big impact).
# Convert to shift: 1 - cos_sim, so higher = more impact from deletion.
print("\nComputing statistics (shift = 1 - cosine_similarity)...")
all_results = []
for token_id, cos_sims in all_gene_shifts.items():
    ens_id = token_to_ensembl.get(token_id, "unknown")
    gene_name = ensembl_to_name.get(ens_id, ens_id)
    shifts = [1.0 - cs for cs in cos_sims]
    all_results.append({
        "token_id": token_id,
        "Ensembl_ID": ens_id,
        "Gene_name": gene_name,
        "mean_cos_sim": np.mean(cos_sims),
        "mean_shift": np.mean(shifts),
        "median_shift": np.median(shifts),
        "std_shift": np.std(shifts),
        "n_cells": len(shifts),
    })

all_df = pd.DataFrame(all_results).sort_values("mean_shift", ascending=False)

# ── Compute statistics for liver genes ───────────────────────────────
liver_results = []
for token_id, cos_sims in gene_shifts.items():
    gene_name = liver_token_ids[token_id]
    ens_id = token_to_ensembl.get(token_id, "unknown")
    shifts = [1.0 - cs for cs in cos_sims]
    liver_results.append({
        "Gene_symbol": gene_name,
        "Ensembl_ID": ens_id,
        "mean_cos_sim": np.mean(cos_sims),
        "mean_shift": np.mean(shifts),
        "median_shift": np.median(shifts),
        "std_shift": np.std(shifts),
        "n_cells": len(shifts),
    })

liver_df = pd.DataFrame(liver_results).sort_values("mean_shift", ascending=False)

# ── Print results ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TOP 20 GENES OVERALL (by mean embedding shift = 1 - cos_sim)")
print(f"{'='*70}")
print(all_df[["Gene_name", "Ensembl_ID", "mean_shift", "mean_cos_sim", "n_cells"]].head(20).to_string(index=False))

# Compute percentile rank for liver genes among all genes
all_shifts = all_df["mean_shift"].values
liver_df["percentile"] = liver_df["mean_shift"].apply(
    lambda x: (all_shifts < x).sum() / len(all_shifts) * 100
)

print(f"\n{'='*70}")
print(f"LIVER CANDIDATE GENES RANKED BY IMPACT (higher shift = more impact)")
print(f"{'='*70}")
print(liver_df[["Gene_symbol", "mean_shift", "mean_cos_sim", "percentile", "n_cells"]].to_string(index=False))

# ── Pathway annotation ───────────────────────────────────────────────
pathway_map = {}
for gene in ["HNF4A", "HNF1A", "HNF1B", "FOXA1", "FOXA2", "FOXA3"]:
    pathway_map[gene] = "Hepatocyte TF"
for gene in ["TEAD1", "TEAD2", "TEAD3", "TEAD4", "YAP1", "WWTR1"]:
    pathway_map[gene] = "Hippo/YAP"
for gene in ["KRAS", "HRAS", "NRAS", "BRAF", "RAF1"]:
    pathway_map[gene] = "RAS/MAPK"
for gene in ["PPARA", "PPARG", "PPARGC1A", "SREBF1", "SREBF2", "NR1H4", "NR1H3"]:
    pathway_map[gene] = "Lipid/MASLD"
for gene in ["CEBPA", "CEBPB", "TBX3", "GATA4", "GATA6", "SOX9", "PROX1"]:
    pathway_map[gene] = "Differentiation"
for gene in ["TGFB1", "SMAD2", "SMAD3", "SMAD4", "SNAI1", "SNAI2", "TWIST1"]:
    pathway_map[gene] = "Fibrosis/TGF-b"
for gene in ["RELA", "NFKB1", "NFKB2", "STAT3", "JUN", "FOS", "TNF", "IL6"]:
    pathway_map[gene] = "Inflammation"
for gene in ["NOTCH1", "NOTCH2", "HES1", "JAG1"]:
    pathway_map[gene] = "Notch"
for gene in ["NFE2L2", "HIF1A", "TP53"]:
    pathway_map[gene] = "Stress"
for gene in ["CTNNB1", "APC", "AXIN2"]:
    pathway_map[gene] = "Wnt"
for gene in ["ACTA2", "COL1A1", "COL3A1", "PDGFRB"]:
    pathway_map[gene] = "Fibrosis marker"
for gene in ["ALB", "AFP", "APOB", "CYP3A4", "CYP1A2"]:
    pathway_map[gene] = "Liver function"

liver_df["Pathway"] = liver_df["Gene_symbol"].map(pathway_map)

# ── Pathway summary ──────────────────────────────────────────────────
pathway_summary = (
    liver_df.groupby("Pathway")["mean_shift"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=False)
)
print(f"\n{'='*70}")
print("PATHWAY-LEVEL IMPACT SUMMARY")
print(f"{'='*70}")
print(pathway_summary.to_string())

# ── Plot 1: Bar chart of liver gene impacts ──────────────────────────
pathway_colors = {
    "Hepatocyte TF": "#E74C3C",
    "Hippo/YAP": "#E67E22",
    "RAS/MAPK": "#F1C40F",
    "Lipid/MASLD": "#2ECC71",
    "Differentiation": "#1ABC9C",
    "Fibrosis/TGF-b": "#3498DB",
    "Inflammation": "#9B59B6",
    "Notch": "#E91E63",
    "Stress": "#795548",
    "Wnt": "#607D8B",
    "Fibrosis marker": "#FF5722",
    "Liver function": "#00BCD4",
}

plot_df = liver_df.sort_values("mean_shift", ascending=True).copy()

fig, ax = plt.subplots(figsize=(10, 16))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

colors = [pathway_colors.get(p, "#999999") for p in plot_df["Pathway"]]
ax.barh(range(len(plot_df)), plot_df["mean_shift"].values, color=colors, alpha=0.8)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df["Gene_symbol"].values, fontsize=8)
ax.set_xlabel("Mean embedding shift (1 - cosine similarity)")
ax.set_title("Impact of liver TF/signaling gene deletion\n(Geneformer V2-104M, zero-shot, 2000 cells)")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=p, alpha=0.8) for p, c in pathway_colors.items()]
ax.legend(handles=legend_elements, loc="lower right", fontsize=7, ncol=2)

fig.savefig(f"{OUTPUT_DIR}/liver_tf_perturbation_barplot.png",
            dpi=150, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"\nBar plot saved to {OUTPUT_DIR}/liver_tf_perturbation_barplot.png")

# ── Plot 2: Pathway-level summary ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

pw_colors = [pathway_colors.get(p, "#999999") for p in pathway_summary.index]
ax.barh(range(len(pathway_summary)), pathway_summary["mean"].values,
        xerr=pathway_summary["std"].values,
        color=pw_colors, alpha=0.8, capsize=3)
ax.set_yticks(range(len(pathway_summary)))
ax.set_yticklabels(pathway_summary.index, fontsize=10)
ax.set_xlabel("Mean embedding shift (1 - cosine similarity)")
ax.set_title("Pathway-level impact of gene deletion in liver cells")

fig.savefig(f"{OUTPUT_DIR}/liver_pathway_summary.png",
            dpi=150, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"Pathway summary saved to {OUTPUT_DIR}/liver_pathway_summary.png")

# ── Plot 3: Liver genes in context of all genes ──────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

ax.hist(all_df["mean_shift"].values, bins=100, alpha=0.4, color="gray", label="All genes")

# Overlay liver genes
for _, row in liver_df.iterrows():
    color = pathway_colors.get(row["Pathway"], "#999999")
    ax.axvline(row["mean_shift"], color=color, alpha=0.6, linewidth=1.5)

# Label top 10 liver genes
top_liver = liver_df.head(10)
for _, row in top_liver.iterrows():
    color = pathway_colors.get(row["Pathway"], "#999999")
    ax.annotate(row["Gene_symbol"],
                xy=(row["mean_shift"], 0),
                xytext=(row["mean_shift"], ax.get_ylim()[1] * 0.7),
                rotation=45, fontsize=7, color=color,
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5))

ax.set_xlabel("Mean embedding shift (1 - cosine similarity)")
ax.set_ylabel("Gene count")
ax.set_title("Distribution of gene deletion impact (all genes vs. liver candidates)")
ax.legend()

fig.savefig(f"{OUTPUT_DIR}/liver_genes_vs_all_distribution.png",
            dpi=150, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"Distribution plot saved to {OUTPUT_DIR}/liver_genes_vs_all_distribution.png")

# ── Save results ─────────────────────────────────────────────────────
liver_df.to_csv(f"{OUTPUT_DIR}/liver_tf_perturbation_results.csv", index=False)
all_df.to_csv(f"{OUTPUT_DIR}/all_genes_perturbation_results.csv", index=False)
print(f"\nResults saved to {OUTPUT_DIR}/")
print("PERTURBATION ANALYSIS COMPLETE!")
