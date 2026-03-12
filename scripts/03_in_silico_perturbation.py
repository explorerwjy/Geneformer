"""
In Silico Perturbation: Gene deletion to shift DCM -> NF
Based on examples/in_silico_perturbation.ipynb
Uses the fine-tuned cell classifier from 01_cell_classification.py
"""
import matplotlib
matplotlib.use("Agg")
import datetime
import glob
import os
from geneformer import EmbExtractor, InSilicoPerturber, InSilicoPerturberStats

# Paths
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset"

# Find the fine-tuned cell classifier model from step 1
cell_class_dirs = sorted(glob.glob("/home/jw3514/Work/Geneformer/outputs/cell_classification/*/"))
if not cell_class_dirs:
    raise FileNotFoundError("No cell classification output found. Run 01_cell_classification.py first.")
latest_cell_class_dir = cell_class_dirs[-1]

# Find the ksplit1 model directory
model_candidates = glob.glob(f"{latest_cell_class_dir}/*cellClassifier*/ksplit1/")
if not model_candidates:
    raise FileNotFoundError(f"No fine-tuned model found in {latest_cell_class_dir}")
FINETUNED_MODEL = model_candidates[0]

print(f"Using fine-tuned model: {FINETUNED_MODEL}")

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"

output_dir = f"/home/jw3514/Work/Geneformer/outputs/in_silico_perturbation/{datestamp}"
os.makedirs(output_dir, exist_ok=True)
output_prefix = "isp_dcm_to_nf"

# Define cell states
cell_states_to_model = {
    "state_key": "disease",
    "start_state": "dcm",
    "goal_state": "nf",
    "alt_states": ["hcm"],
}
filter_data_dict = {"cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]}

# --- Step 1: Extract state embeddings ---
print("=" * 60)
print("STEP 1: Extracting state embeddings")
print("=" * 60)

embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=3,
    filter_data=filter_data_dict,
    max_ncells=1000,
    emb_layer=0,
    summary_stat="exact_mean",
    forward_batch_size=256,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    model_version="V1",
    nproc=10,
)

state_embs_dict = embex.get_state_embs(
    cell_states_to_model,
    FINETUNED_MODEL,
    INPUT_DATA,
    output_dir,
    output_prefix,
)
print("State embeddings extracted!")

# --- Step 2: Run in silico perturbation ---
print("=" * 60)
print("STEP 2: Running in silico perturbation (gene deletions)")
print("  NOTE: This perturbs top 50 genes to keep runtime reasonable.")
print("  Set genes_to_perturb='all' for full analysis.")
print("=" * 60)

isp = InSilicoPerturber(
    perturb_type="delete",
    perturb_rank_shift=None,
    genes_to_perturb="all",
    combos=0,
    anchor_gene=None,
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data=filter_data_dict,
    cell_states_to_model=cell_states_to_model,
    state_embs_dict=state_embs_dict,
    max_ncells=200,  # reduced for faster runtime
    emb_layer=0,
    forward_batch_size=400,
    model_version="V1",
    nproc=10,
)

isp.perturb_data(
    FINETUNED_MODEL,
    INPUT_DATA,
    output_dir,
    output_prefix,
)
print("Perturbation complete!")

# --- Step 3: Compute statistics ---
print("=" * 60)
print("STEP 3: Computing perturbation statistics")
print("=" * 60)

ispstats = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed="all",
    combos=0,
    anchor_gene=None,
    cell_states_to_model=cell_states_to_model,
    model_version="V1",
)

ispstats.get_stats(
    output_dir,
    None,
    output_dir,
    output_prefix,
)

print(f"\nAll outputs saved to: {output_dir}")
print("IN SILICO PERTURBATION COMPLETE!")
