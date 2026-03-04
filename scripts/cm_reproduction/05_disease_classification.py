# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Disease Classification: NF vs HCM vs DCM Cardiomyocytes
#
# Fine-tune Geneformer V2-316M to classify cardiomyocyte disease states
# (NF = non-failing, HCM = hypertrophic cardiomyopathy, DCM = dilated cardiomyopathy).
#
# Uses patient-level splitting (by donor_id) to avoid data leakage.

# %%
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CLI execution

import datetime
import glob
import os
import pickle

import numpy as np
from geneformer import Classifier

# %% [markdown]
# ## Configuration

# %%
# Paths
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V2-316M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/Geneformer/data/tokenized/chaffin_cardiomyocytes.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "disease_classifier"
output_dir = f"/home/jw3514/Work/Geneformer/Geneformer/outputs/disease_classification/{datestamp}"
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## Patient-level train/eval/test split
#
# Donors grouped by disease, then split ~70/15/15 within each group
# to ensure all three disease classes are represented in each split.

# %%
# Donors by disease group (from data inspection)
dcm_donors = ["P1290", "P1300", "P1304", "P1358", "P1371", "P1430", "P1437",
               "P1472", "P1504", "P1606", "P1617"]  # 11 donors
hcm_donors = ["P1422", "P1425", "P1447", "P1462", "P1479", "P1508", "P1510",
               "P1602", "P1630", "P1631", "P1685", "P1707", "P1722", "P1726",
               "P1735"]  # 15 donors
nf_donors = ["P1515", "P1516", "P1539", "P1540", "P1547", "P1549", "P1558",
              "P1561", "P1582", "P1600", "P1603", "P1610", "P1622", "P1678",
              "P1702", "P1718"]  # 16 donors

# Split each group ~70/15/15
# DCM: 8 train / 1 eval / 2 test
dcm_train = ["P1290", "P1300", "P1304", "P1358", "P1371", "P1430", "P1472", "P1504"]
dcm_eval = ["P1606"]
dcm_test = ["P1437", "P1617"]

# HCM: 11 train / 2 eval / 2 test
hcm_train = ["P1422", "P1425", "P1447", "P1462", "P1479", "P1508", "P1602",
              "P1630", "P1631", "P1707", "P1722"]
hcm_eval = ["P1510", "P1726"]
hcm_test = ["P1685", "P1735"]

# NF: 11 train / 2 eval / 3 test
nf_train = ["P1515", "P1539", "P1540", "P1547", "P1549", "P1558", "P1561",
             "P1582", "P1600", "P1603", "P1610"]
nf_eval = ["P1622", "P1678"]
nf_test = ["P1516", "P1702", "P1718"]

train_ids = dcm_train + hcm_train + nf_train
eval_ids = dcm_eval + hcm_eval + nf_eval
test_ids = dcm_test + hcm_test + nf_test

print(f"Train donors: {len(train_ids)} ({len(dcm_train)} DCM, {len(hcm_train)} HCM, {len(nf_train)} NF)")
print(f"Eval donors:  {len(eval_ids)} ({len(dcm_eval)} DCM, {len(hcm_eval)} HCM, {len(nf_eval)} NF)")
print(f"Test donors:  {len(test_ids)} ({len(dcm_test)} DCM, {len(hcm_test)} HCM, {len(nf_test)} NF)")
print(f"Total donors: {len(train_ids) + len(eval_ids) + len(test_ids)}")

# %% [markdown]
# ## Step 1: Initialize classifier and prepare data

# %%
training_args = {
    "num_train_epochs": 1,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 8,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "seed": 42,
}

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    filter_data=None,  # all cells are already cardiomyocytes
    training_args=training_args,
    freeze_layers=8,  # V2-316M has 12 layers; freeze bottom 8
    num_crossval_splits=1,
    forward_batch_size=100,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

# %%
print("=" * 60)
print("STEP 1: Preparing data (train+eval vs test split)")
print("=" * 60)

train_test_split = {
    "attr_key": "donor_id",
    "train": train_ids + eval_ids,
    "test": test_ids,
}

cc.prepare_data(
    input_data_file=INPUT_DATA,
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_test_split,
)
print("Data prepared successfully!")

# %% [markdown]
# ## Step 2: Train model (with train/eval patient split)

# %%
print("=" * 60)
print("STEP 2: Training with patient-level train/eval split")
print("=" * 60)

train_valid_split = {
    "attr_key": "donor_id",
    "train": train_ids,
    "eval": eval_ids,
}

all_metrics = cc.validate(
    model_directory=MODEL_DIR,
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_valid_split,
)

print(f"\nValidation metrics:")
# validate() returns lists (one value per fold); with num_crossval_splits=1, take index 0
print(f"  Macro F1: {all_metrics['macro_f1'][0]:.4f}")
print(f"  Accuracy: {all_metrics['acc'][0]:.4f}")
print(f"  Confusion matrix:\n{all_metrics['conf_matrix']}")

# %% [markdown]
# ## Step 3: Evaluate on held-out test set

# %%
print("=" * 60)
print("STEP 3: Evaluating on held-out test set")
print("=" * 60)

# Create a fresh Classifier for evaluation (no training args needed)
cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    forward_batch_size=100,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

# Find the saved fine-tuned model
saved_model_candidates = glob.glob(f"{output_dir}/*cellClassifier*/ksplit1/")
if saved_model_candidates:
    saved_model_dir = saved_model_candidates[0]
else:
    raise FileNotFoundError(
        f"No saved model found in {output_dir}. Check training output."
    )

print(f"Using saved model: {saved_model_dir}")

all_metrics_test = cc_eval.evaluate_saved_model(
    model_directory=saved_model_dir,
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

print(f"\nTest metrics:")
print(f"  Macro F1: {all_metrics_test['macro_f1']:.4f}")
print(f"  Accuracy: {all_metrics_test['acc']:.4f}")
print(f"  Confusion matrix:\n{all_metrics_test['conf_matrix']}")

# %% [markdown]
# ## Step 4: Plot confusion matrix and predictions

# %%
print("=" * 60)
print("STEP 4: Plotting results")
print("=" * 60)

cc_eval.plot_conf_mat(
    conf_mat_dict={"Geneformer_V2_316M": all_metrics_test["conf_matrix"]},
    output_directory=output_dir,
    output_prefix=output_prefix,
    custom_class_order=["NF", "HCM", "DCM"],
)

# Find the predictions file
pred_file_candidates = glob.glob(f"{output_dir}/{output_prefix}_pred_dict.pkl")
if pred_file_candidates:
    pred_file = pred_file_candidates[0]
else:
    # Try alternative location
    pred_file_candidates = glob.glob(f"{output_dir}/*pred_dict*.pkl")
    if pred_file_candidates:
        pred_file = pred_file_candidates[0]
    else:
        pred_file = None
        print("Warning: Predictions file not found; skipping prediction plot.")

if pred_file is not None:
    print(f"Using predictions file: {pred_file}")
    cc_eval.plot_predictions(
        predictions_file=pred_file,
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        title="Disease State",
        output_directory=output_dir,
        output_prefix=output_prefix,
        custom_class_order=["NF", "HCM", "DCM"],
    )

# %% [markdown]
# ## Summary

# %%
print("=" * 60)
print("DISEASE CLASSIFICATION COMPLETE")
print("=" * 60)
print(f"\nAll outputs saved to: {output_dir}")
print(f"\nTest set results:")
print(f"  Macro F1: {all_metrics_test['macro_f1']:.4f}")
print(f"  Accuracy: {all_metrics_test['acc']:.4f}")
print(f"  Confusion matrix:\n{all_metrics_test['conf_matrix']}")
