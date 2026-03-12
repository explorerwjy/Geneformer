"""
Step 5: In silico perturbation of liver-relevant TFs and signaling genes.

Deletes each candidate gene individually from cell embeddings and measures
the cosine shift. Uses the pretrained Geneformer V2-104M model (zero-shot).

Two analyses:
  A) All cells together — which TFs most impact the global transcriptome?
  B) Per disease group — do different TFs matter in different conditions?
"""

import datetime
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from geneformer import InSilicoPerturber, InSilicoPerturberStats

# ── Paths ───────────────────────────────────────────────────────────
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-104M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/QiuyanLiver/tokenized/liver_qiuyan.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
OUTPUT_DIR = f"/home/jw3514/Work/Geneformer/outputs/liver_perturbation/{datestamp}"
ISP_OUTPUT_DIR = f"{OUTPUT_DIR}/isp_raw"
STATS_OUTPUT_DIR = f"{OUTPUT_DIR}/stats"

for d in [ISP_OUTPUT_DIR, STATS_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Curated liver TF / signaling gene list (Ensembl IDs) ────────────
# All verified present in Geneformer V2-104M token dictionary
LIVER_GENES = {
    # Hepatocyte master TFs
    "HNF4A": "ENSG00000101076",
    "HNF1A": "ENSG00000135100",
    "HNF1B": "ENSG00000275410",
    "FOXA1": "ENSG00000129514",
    "FOXA2": "ENSG00000125798",
    "FOXA3": "ENSG00000170608",
    # Hippo/YAP pathway
    "TEAD1": "ENSG00000187079",
    "TEAD2": "ENSG00000074219",
    "TEAD3": "ENSG00000007866",
    "TEAD4": "ENSG00000197905",
    "YAP1": "ENSG00000137693",
    "WWTR1": "ENSG00000018408",  # TAZ
    # RAS/MAPK
    "KRAS": "ENSG00000133703",
    "HRAS": "ENSG00000174775",
    "NRAS": "ENSG00000213281",
    "BRAF": "ENSG00000157764",
    "RAF1": "ENSG00000132155",
    # Lipid metabolism / MASLD
    "PPARA": "ENSG00000186951",
    "PPARG": "ENSG00000132170",
    "PPARGC1A": "ENSG00000109819",
    "SREBF1": "ENSG00000072310",
    "SREBF2": "ENSG00000198911",
    "NR1H4": "ENSG00000012504",   # FXR
    "NR1H3": "ENSG00000025434",   # LXR
    # Liver differentiation
    "CEBPA": "ENSG00000245848",
    "CEBPB": "ENSG00000172216",
    "TBX3": "ENSG00000135111",
    "GATA4": "ENSG00000136574",
    "GATA6": "ENSG00000141448",
    "SOX9": "ENSG00000125398",
    "PROX1": "ENSG00000117707",
    # Fibrosis / TGF-beta
    "TGFB1": "ENSG00000105329",
    "SMAD2": "ENSG00000175387",
    "SMAD3": "ENSG00000166949",
    "SMAD4": "ENSG00000141646",
    "SNAI1": "ENSG00000124216",
    "SNAI2": "ENSG00000019549",
    "TWIST1": "ENSG00000122691",
    # Inflammation
    "RELA": "ENSG00000173039",
    "NFKB1": "ENSG00000109320",
    "NFKB2": "ENSG00000077150",
    "STAT3": "ENSG00000168610",
    "JUN": "ENSG00000177606",
    "FOS": "ENSG00000170345",
    "TNF": "ENSG00000232810",
    "IL6": "ENSG00000136244",
    # Notch pathway
    "NOTCH1": "ENSG00000148400",
    "NOTCH2": "ENSG00000134250",
    "HES1": "ENSG00000114315",
    "JAG1": "ENSG00000101384",
    # Oxidative stress / hypoxia
    "NFE2L2": "ENSG00000116044",  # NRF2
    "HIF1A": "ENSG00000100644",
    "TP53": "ENSG00000141510",
    # Wnt pathway
    "CTNNB1": "ENSG00000168036",
    "APC": "ENSG00000134982",
    "AXIN2": "ENSG00000168646",
    # Fibrosis markers
    "ACTA2": "ENSG00000107796",
    "COL1A1": "ENSG00000108821",
    "COL3A1": "ENSG00000168542",
    "PDGFRB": "ENSG00000113721",
    # Liver function genes
    "ALB": "ENSG00000163631",
    "AFP": "ENSG00000081051",
    "APOB": "ENSG00000084674",
    "CYP3A4": "ENSG00000160868",
    "CYP1A2": "ENSG00000140505",
}

genes_to_perturb = list(LIVER_GENES.values())
gene_name_lookup = {v: k for k, v in LIVER_GENES.items()}

print(f"Perturbing {len(genes_to_perturb)} liver-relevant genes")
print(f"Model: Geneformer V2-104M (pretrained, zero-shot)")
print(f"Dataset: {INPUT_DATA}")

# ── Configuration ───────────────────────────────────────────────────
MAX_NCELLS = None  # Use all 166K cells
FORWARD_BATCH_SIZE = 50

# ── Step 1: Run in silico perturbation (all genes) ──────────────────
# Note: genes_to_perturb="all" is required because the delete filter
# requires ALL listed genes to be present in each cell (too strict).
# We perturb all genes, then filter to our candidates in post-processing.
print("\n" + "=" * 60)
print("STEP 1: In silico gene deletion (all genes, 2000 cells)")
print("=" * 60)

isp = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb="all",
    combos=0,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    max_ncells=MAX_NCELLS,
    emb_layer=-1,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version="V2",
    nproc=1,  # avoid multiprocessing spawn issues
)

isp.perturb_data(
    model_directory=MODEL_DIR,
    input_data_file=INPUT_DATA,
    output_directory=ISP_OUTPUT_DIR,
    output_prefix="liver_tfs",
)

print("\nPerturbation data saved. Run 05b_analyze_perturbation.py for analysis.")
print(f"Raw output in: {ISP_OUTPUT_DIR}")
print("IN SILICO PERTURBATION COMPLETE!")
