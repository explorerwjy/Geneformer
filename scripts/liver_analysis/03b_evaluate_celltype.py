"""
Step 3b: Evaluate the trained cell type classifier on held-out test donors.
Also plots confusion matrix and per-class metrics.
"""

import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
from geneformer import Classifier

# ── Paths ───────────────────────────────────────────────────────────
OUTPUT_BASE = "/home/jw3514/Work/Geneformer/outputs/liver_celltype/260310190023"
MODEL_DIR = f"{OUTPUT_BASE}/260310_geneformer_cellClassifier_celltype/ksplit1"
ID_CLASS_DICT = f"{OUTPUT_BASE}/celltype_id_class_dict.pkl"
TEST_DATA = f"{OUTPUT_BASE}/celltype_labeled_test.dataset"

# ── Load test data and model ────────────────────────────────────────
print("Loading id_class_dict...")
with open(ID_CLASS_DICT, "rb") as f:
    id_class_dict = pickle.load(f)
print(f"  Classes: {id_class_dict}")

print("Loading test dataset...")
test_ds = load_from_disk(TEST_DATA)
print(f"  Test cells: {len(test_ds)}")

# ── Run evaluation ──────────────────────────────────────────────────
print("\nEvaluating on test set...")

cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_type", "states": "all"},
    forward_batch_size=50,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

test_metrics = cc_eval.evaluate_saved_model(
    model_directory=MODEL_DIR,
    id_class_dict_file=ID_CLASS_DICT,
    test_data_file=TEST_DATA,
    output_directory=OUTPUT_BASE,
    output_prefix="celltype_test",
)

print(f"\n{'='*60}")
print("TEST SET RESULTS")
print(f"{'='*60}")
print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
print(f"  Accuracy: {test_metrics['acc']:.4f}")

# Print confusion matrix
print(f"\nConfusion matrix:\n{test_metrics['conf_matrix']}")

# Save summary
with open(f"{OUTPUT_BASE}/celltype_test_summary.txt", "w") as f:
    f.write(f"Cell Type Classification - Test Set Results\n")
    f.write(f"Model: Geneformer V2-104M, 1 epoch, freeze_layers=4\n")
    f.write(f"Test donors: S14 (Healthy), S9 (Alcohol Hepatitis)\n")
    f.write(f"Test cells: {len(test_ds)}\n")
    f.write(f"Accuracy: {test_metrics['acc']:.4f}\n")
    f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")

print(f"\nAll outputs saved to: {OUTPUT_BASE}")
print("EVALUATION COMPLETE!")
