"""
Extract and Plot Cell Embeddings
Based on examples/extract_and_plot_cell_embeddings.ipynb
Uses the fine-tuned cell classifier from 01_cell_classification.py
"""
import matplotlib
matplotlib.use("Agg")
import datetime
import glob
import os
from geneformer import EmbExtractor

# Paths
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset"

# Find the fine-tuned cell classifier model from step 1
cell_class_dirs = sorted(glob.glob("/home/jw3514/Work/Geneformer/outputs/cell_classification/*/"))
if not cell_class_dirs:
    raise FileNotFoundError("No cell classification output found. Run 01_cell_classification.py first.")
latest_cell_class_dir = cell_class_dirs[-1]

model_candidates = glob.glob(f"{latest_cell_class_dir}/*cellClassifier*/ksplit1/")
if not model_candidates:
    raise FileNotFoundError(f"No fine-tuned model found in {latest_cell_class_dir}")
FINETUNED_MODEL = model_candidates[0]

print(f"Using fine-tuned model: {FINETUNED_MODEL}")

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"

output_dir = f"/home/jw3514/Work/Geneformer/outputs/embeddings/{datestamp}"
os.makedirs(output_dir, exist_ok=True)
output_prefix = "cardiomyopathy_embs"

# --- Step 1: Extract embeddings ---
print("=" * 60)
print("STEP 1: Extracting cell embeddings")
print("=" * 60)

embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=3,
    filter_data={"cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]},
    max_ncells=1000,
    emb_layer=0,
    emb_label=["disease", "cell_type"],
    labels_to_plot=["disease"],
    forward_batch_size=200,
    model_version="V1",
    nproc=10,
)

embs = embex.extract_embs(
    FINETUNED_MODEL,
    INPUT_DATA,
    output_dir,
    output_prefix,
)

print(f"Embeddings shape: {embs.shape}")
print(f"Columns: {list(embs.columns[:5])}...{list(embs.columns[-5:])}")

# --- Step 2: Plot UMAP ---
print("=" * 60)
print("STEP 2: Plotting UMAP of cell embeddings")
print("=" * 60)

embex.plot_embs(
    embs=embs,
    plot_style="umap",
    output_directory=output_dir,
    output_prefix=f"{output_prefix}_umap",
)
print("UMAP saved!")

# --- Step 3: Plot heatmap ---
print("=" * 60)
print("STEP 3: Plotting heatmap of cell embeddings")
print("=" * 60)

embex.plot_embs(
    embs=embs,
    plot_style="heatmap",
    output_directory=output_dir,
    output_prefix=f"{output_prefix}_heatmap",
)
print("Heatmap saved!")

print(f"\nAll outputs saved to: {output_dir}")
print("EMBEDDING EXTRACTION COMPLETE!")
