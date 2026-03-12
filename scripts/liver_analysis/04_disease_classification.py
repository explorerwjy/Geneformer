"""
Step 4: Fine-tune Geneformer for disease classification on liver data.
Classifies cells into: Healthy, Alcohol Cirrhosis, Alcohol Hepatitis,
MASH Cirrhosis, MASLD, MASH Fibrosis.

Can be run on all cells or restricted to specific cell types (e.g., hepatocytes only).
Uses patient-level splitting to avoid data leakage.
"""

import datetime
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from datasets import load_from_disk

from geneformer import Classifier

# ── Configuration ───────────────────────────────────────────────────
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-104M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/QiuyanLiver/tokenized/liver_qiuyan.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
output_dir = f"/home/jw3514/Work/Geneformer/outputs/liver_disease/{datestamp}"
os.makedirs(output_dir, exist_ok=True)
output_prefix = "disease"

# ── Add disease label to dataset ────────────────────────────────────
print("Adding disease labels to tokenized dataset...")
ds = load_from_disk(INPUT_DATA)

disease_map = {
    "S1": "Alcohol_Cirrhosis", "S2": "Alcohol_Cirrhosis",
    "S3": "Alcohol_Cirrhosis", "S4": "Alcohol_Cirrhosis",
    "S5": "Alcohol_Hepatitis", "S6": "Alcohol_Hepatitis",
    "S7": "Alcohol_Hepatitis", "S8": "Alcohol_Hepatitis",
    "S9": "Alcohol_Hepatitis",
    "S12": "Healthy", "S13": "Healthy", "S14": "Healthy",
    "S23": "Healthy", "S27": "Healthy", "S28": "Healthy",
    "S15": "MASH_Cirrhosis", "S16": "MASH_Cirrhosis",
    "S17": "MASH_Cirrhosis", "S18": "MASH_Cirrhosis",
    "S19": "MASLD", "S20": "MASLD", "S21": "MASLD",
    "S33": "MASH_Fibrosis", "S35": "MASH_Fibrosis",
    "S36": "MASH_Fibrosis", "S38": "MASH_Fibrosis",
}

ds = ds.map(lambda x: {"disease": disease_map[x["sample_id"]]}, num_proc=10)

# Save augmented dataset
augmented_path = INPUT_DATA.replace(".dataset", "_with_disease.dataset")
ds.save_to_disk(augmented_path)
print(f"Saved augmented dataset to {augmented_path}")
print(f"Disease distribution:")
from collections import Counter
print(Counter(ds["disease"]))

# ── Patient-level train/eval/test split ─────────────────────────────
# ~70/15/15 within each disease group
train_ids = [
    "S12", "S13", "S23", "S27",   # Healthy (4)
    "S1", "S2", "S3",              # Alcohol Cirrhosis (3)
    "S5", "S6", "S7",              # Alcohol Hepatitis (3)
    "S15", "S16", "S17",           # MASH Cirrhosis (3)
    "S19", "S20",                  # MASLD (2)
    "S33", "S35", "S36",           # MASH Fibrosis (3)
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
    "S9",    # Alcohol Hepatitis
]

all_train_eval = train_ids + eval_ids
print(f"\nTrain: {len(train_ids)}, Eval: {len(eval_ids)}, Test: {len(test_ids)} donors")

# ── Focus on hepatocytes (most abundant, most relevant for liver disease) ──
# Filtering to hepatocytes makes the task more meaningful biologically
FILTER_CELLTYPE = None  # Set to {"cell_type": ["Hepatocytes"]} to restrict

# ── Training ────────────────────────────────────────────────────────
training_args = {
    "num_train_epochs": 1,
    "learning_rate": 2e-5,
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
    cell_state_dict={"state_key": "disease", "states": "all"},
    filter_data=FILTER_CELLTYPE,
    training_args=training_args,
    freeze_layers=4,
    num_crossval_splits=1,
    forward_batch_size=50,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

print("=" * 60)
print("STEP 1: Preparing data")
print("=" * 60)

cc.prepare_data(
    input_data_file=augmented_path,
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict={
        "attr_key": "sample_id",
        "train": all_train_eval,
        "test": test_ids,
    },
)
print("Data prepared!")

print("=" * 60)
print("STEP 2: Training disease classifier")
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

# ── Evaluate on test set ────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Test set evaluation")
print("=" * 60)

cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    forward_batch_size=50,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

saved_model = glob.glob(f"{output_dir}/*cellClassifier*/ksplit1/")
if saved_model:
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

    cc_eval.plot_conf_mat(
        conf_mat_dict={"Geneformer_V2_104M": test_metrics["conf_matrix"]},
        output_directory=output_dir,
        output_prefix=output_prefix,
    )

print(f"\nAll outputs saved to: {output_dir}")
print("DISEASE CLASSIFICATION COMPLETE!")
