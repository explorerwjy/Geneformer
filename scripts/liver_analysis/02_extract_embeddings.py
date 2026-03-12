"""
Step 2: Extract cell embeddings from pretrained Geneformer (no fine-tuning).
Fast — just forward pass through the model.
Produces UMAP plots colored by cell type and disease condition.
"""

import datetime
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from geneformer import EmbExtractor

# ── Paths ───────────────────────────────────────────────────────────
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-104M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/QiuyanLiver/tokenized/liver_qiuyan.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
OUTPUT_DIR = f"/home/jw3514/Work/Geneformer/outputs/liver_embeddings/{datestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Extract embeddings ──────────────────────────────────────────────
print("=" * 60)
print("Extracting cell embeddings (pretrained Geneformer V2-104M)")
print("=" * 60)

embex = EmbExtractor(
    model_type="Pretrained",
    num_classes=0,
    max_ncells=None,  # all cells
    emb_layer=-1,  # last hidden layer
    emb_label=["cell_type", "sample_id"],
    labels_to_plot=["cell_type", "sample_id"],
    forward_batch_size=50,
    model_version="V2",
    nproc=10,
)

embs = embex.extract_embs(
    model_directory=MODEL_DIR,
    input_data_file=INPUT_DATA,
    output_directory=OUTPUT_DIR,
    output_prefix="liver_embs",
)

print(f"\nEmbeddings shape: {embs.shape}")
print(f"Columns: {list(embs.columns[:3])}...{list(embs.columns[-3:])}")

# ── Plot UMAPs ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Plotting UMAP by cell type")
print("=" * 60)

embex.plot_embs(
    embs=embs,
    plot_style="umap",
    output_directory=OUTPUT_DIR,
    output_prefix="liver_umap_celltype",
)

# Also add disease condition to embeddings for a second UMAP
# Map sample_id -> disease condition
disease_map = {
    "S1": "Alcohol Cirrhosis", "S2": "Alcohol Cirrhosis",
    "S3": "Alcohol Cirrhosis", "S4": "Alcohol Cirrhosis",
    "S5": "Alcohol Hepatitis", "S6": "Alcohol Hepatitis",
    "S7": "Alcohol Hepatitis", "S8": "Alcohol Hepatitis",
    "S9": "Alcohol Hepatitis",
    "S12": "Healthy", "S13": "Healthy", "S14": "Healthy",
    "S23": "Healthy", "S27": "Healthy", "S28": "Healthy",
    "S15": "MASH Cirrhosis", "S16": "MASH Cirrhosis",
    "S17": "MASH Cirrhosis", "S18": "MASH Cirrhosis",
    "S19": "MASLD", "S20": "MASLD", "S21": "MASLD",
    "S33": "MASH Fibrosis", "S35": "MASH Fibrosis",
    "S36": "MASH Fibrosis", "S38": "MASH Fibrosis",
}
embs["disease"] = embs["sample_id"].map(disease_map)

# Save embeddings with disease labels
embs.to_csv(os.path.join(OUTPUT_DIR, "liver_embeddings_with_metadata.csv"))
print(f"\nEmbeddings with metadata saved to {OUTPUT_DIR}")

# Custom UMAP colored by disease
print("\nPlotting UMAP by disease condition...")
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

emb_cols = [c for c in embs.columns if c not in ["cell_type", "sample_id", "disease"]]
X = embs[emb_cols].values

print("  Running UMAP...")
reducer = UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=10)
coords = reducer.fit_transform(X)

for label_col, fname in [("cell_type", "umap_celltype"), ("disease", "umap_disease"), ("sample_id", "umap_sample")]:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    categories = sorted(embs[label_col].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    for i, cat in enumerate(categories):
        mask = embs[label_col] == cat
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=cat, s=1, alpha=0.3, rasterized=True)

    ax.legend(markerscale=5, fontsize=8, loc="best", framealpha=0.8)
    ax.set_title(f"Geneformer V2-104M Embeddings — {label_col}", fontsize=14)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    outpath = os.path.join(OUTPUT_DIR, f"{fname}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved {outpath}")

print(f"\nAll outputs in: {OUTPUT_DIR}")
print("DONE!")
