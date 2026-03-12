"""
Step 3: Fine-tune Geneformer for cell type classification on liver data.
Uses patient-level splitting to avoid data leakage.
"""

import datetime
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from geneformer import Classifier

# ── Configuration ───────────────────────────────────────────────────
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-104M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/QiuyanLiver/tokenized/liver_qiuyan.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
output_dir = f"/home/jw3514/Work/Geneformer/outputs/liver_celltype/{datestamp}"
os.makedirs(output_dir, exist_ok=True)
output_prefix = "celltype"

# ── Patient-level train/eval/test split ─────────────────────────────
# Group by disease to ensure balanced representation
# Healthy: S12, S13, S14, S23, S27, S28 (6)
# Alcohol Cirrhosis: S1, S2, S3, S4 (4)
# Alcohol Hepatitis: S5, S6, S7, S8, S9 (5)
# MASH Cirrhosis: S15, S16, S17, S18 (4)
# MASLD: S19, S20, S21 (3)
# MASH Fibrosis: S33, S35, S36, S38 (4)

# ~70/15/15 split within each disease group
train_ids = [
    # Healthy (4 train)
    "S12", "S13", "S23", "S27",
    # Alcohol Cirrhosis (3 train)
    "S1", "S2", "S3",
    # Alcohol Hepatitis (3 train)
    "S5", "S6", "S7",
    # MASH Cirrhosis (3 train)
    "S15", "S16", "S17",
    # MASLD (2 train)
    "S19", "S20",
    # MASH Fibrosis (3 train)
    "S33", "S35", "S36",
]

eval_ids = [
    "S28",   # Healthy
    "S4",    # Alcohol Cirrhosis
    "S8",    # Alcohol Hepatitis
    "S18",   # MASH Cirrhosis
    "S21",   # MASLD
    "S38",   # MASH Fibrosis
]

test_ids = [
    "S14",   # Healthy
    # No Alcohol Cirrhosis test (only 4 donors)
    "S9",    # Alcohol Hepatitis
    # No MASH Cirrhosis test (only 4 donors)
    # S21 already in eval
]

# With limited donors, use simpler 2-way split for Classifier
# train+eval vs test
all_train_eval = train_ids + eval_ids
all_test = test_ids

print(f"Train donors ({len(train_ids)}): {train_ids}")
print(f"Eval donors ({len(eval_ids)}): {eval_ids}")
print(f"Test donors ({len(all_test)}): {all_test}")

# ── Step 1: Initialize classifier ──────────────────────────────────
training_args = {
    "num_train_epochs": 1,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 16,
    "fp16": True,
    "gradient_checkpointing": True,
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "seed": 42,
}

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_type", "states": "all"},
    training_args=training_args,
    freeze_layers=4,  # V2-104M has 6 layers; freeze bottom 4
    num_crossval_splits=1,
    forward_batch_size=50,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

# ── Step 2: Prepare data ───────────────────────────────────────────
print("=" * 60)
print("STEP 1: Preparing data")
print("=" * 60)

cc.prepare_data(
    input_data_file=INPUT_DATA,
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict={
        "attr_key": "sample_id",
        "train": all_train_eval,
        "test": all_test,
    },
)
print("Data prepared!")

# ── Step 3: Train with train/eval split ─────────────────────────────
print("=" * 60)
print("STEP 2: Training cell type classifier")
print("=" * 60)

all_metrics = cc.validate(
    model_directory=MODEL_DIR,
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict={
        "attr_key": "sample_id",
        "train": train_ids,
        "eval": eval_ids,
    },
)

print(f"\nValidation metrics:")
print(f"  Macro F1: {all_metrics['macro_f1'][0]:.4f}")
print(f"  Accuracy: {all_metrics['acc'][0]:.4f}")

# ── Step 4: Evaluate on held-out test set ───────────────────────────
print("=" * 60)
print("STEP 3: Evaluating on held-out test donors")
print("=" * 60)

cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_type", "states": "all"},
    forward_batch_size=50,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

saved_model = glob.glob(f"{output_dir}/*cellClassifier*/ksplit1/")
if not saved_model:
    print("WARNING: No saved model found, skipping test evaluation")
else:
    saved_model_dir = saved_model[0]
    print(f"Using model: {saved_model_dir}")

    test_metrics = cc_eval.evaluate_saved_model(
        model_directory=saved_model_dir,
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )

    print(f"\nTest metrics:")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
    print(f"  Confusion matrix:\n{test_metrics['conf_matrix']}")

    # Plot confusion matrix
    cc_eval.plot_conf_mat(
        conf_mat_dict={"Geneformer_V2_104M": test_metrics["conf_matrix"]},
        output_directory=output_dir,
        output_prefix=output_prefix,
    )

print(f"\nAll outputs saved to: {output_dir}")
print("CELL TYPE CLASSIFICATION COMPLETE!")
