"""
Step 4b: Evaluate the trained disease classifier on held-out test donors.
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from datasets import load_from_disk
from geneformer import Classifier

# ── Paths ───────────────────────────────────────────────────────────
OUTPUT_BASE = "/home/jw3514/Work/Geneformer/outputs/liver_disease/260310220537"
MODEL_DIR = f"{OUTPUT_BASE}/260310_geneformer_cellClassifier_disease/ksplit1"
ID_CLASS_DICT = f"{OUTPUT_BASE}/disease_id_class_dict.pkl"
TEST_DATA = f"{OUTPUT_BASE}/disease_labeled_test.dataset"

# ── Load test data info ──────────────────────────────────────────────
print("Loading id_class_dict...")
with open(ID_CLASS_DICT, "rb") as f:
    id_class_dict = pickle.load(f)
print(f"  Classes: {id_class_dict}")

print("Loading test dataset...")
test_ds = load_from_disk(TEST_DATA)
print(f"  Test cells: {len(test_ds)}")

# Check disease distribution in test set
from collections import Counter
print(f"  Disease distribution: {Counter(test_ds['label'])}")

# ── Run evaluation ──────────────────────────────────────────────────
print("\nEvaluating on test set...")

cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
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
    output_prefix="disease_test",
)

print(f"\n{'='*60}")
print("TEST SET RESULTS")
print(f"{'='*60}")
print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
print(f"  Accuracy: {test_metrics['acc']:.4f}")
print(f"  Confusion matrix:\n{test_metrics['conf_matrix']}")

# Save summary
with open(f"{OUTPUT_BASE}/disease_test_summary.txt", "w") as f:
    f.write(f"Disease Classification - Test Set Results\n")
    f.write(f"Model: Geneformer V2-104M, 1 epoch, freeze_layers=4\n")
    f.write(f"Test donors: S14 (Healthy), S9 (Alcohol Hepatitis)\n")
    f.write(f"Test cells: {len(test_ds)}\n")
    f.write(f"Accuracy: {test_metrics['acc']:.4f}\n")
    f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
    f.write(f"Classes: {id_class_dict}\n")

print(f"\nAll outputs saved to: {OUTPUT_BASE}")
print("DISEASE EVALUATION COMPLETE!")
