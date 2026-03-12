"""
Cell Classification: Cardiomyopathy Disease States (DCM/HCM/NF)
Based on examples/cell_classification.ipynb
Uses V1 model with disease classification dataset.
"""
import datetime
import os
from geneformer import Classifier

# Paths
MODEL_DIR = "/home/jw3514/Work/Geneformer/models/Geneformer/Geneformer-V1-10M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "cm_classifier"
output_dir = f"/home/jw3514/Work/Geneformer/outputs/cell_classification/{datestamp}"
os.makedirs(output_dir, exist_ok=True)

# --- Step 1: Prepare data with train/test split ---
print("=" * 60)
print("STEP 1: Preparing data")
print("=" * 60)

filter_data_dict = {"cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]}

training_args = {
    "num_train_epochs": 0.9,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay": 0.258828,
    "per_device_train_batch_size": 12,
    "seed": 73,
}

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    filter_data=filter_data_dict,
    training_args=training_args,
    max_ncells=None,
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=200,
    model_version="V1",
    nproc=10,
)

# Use same splits as the example notebook
train_ids = ["1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561",
             "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617",
             "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371",
             "1549", "1515"]
eval_ids = ["1422", "1510", "1539", "1606", "1702"]
test_ids = ["1437", "1516", "1602", "1685", "1718"]

train_test_id_split_dict = {
    "attr_key": "individual",
    "train": train_ids + eval_ids,
    "test": test_ids,
}

cc.prepare_data(
    input_data_file=INPUT_DATA,
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_test_id_split_dict,
)
print("Data prepared successfully!")

# --- Step 2: Train with cross-validation ---
print("=" * 60)
print("STEP 2: Training with cross-validation")
print("=" * 60)

train_valid_id_split_dict = {
    "attr_key": "individual",
    "train": train_ids,
    "eval": eval_ids,
}

all_metrics = cc.validate(
    model_directory=MODEL_DIR,
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_valid_id_split_dict,
)

print(f"\nTraining metrics:")
print(f"  Macro F1: {all_metrics['macro_f1']}")
print(f"  Accuracy: {all_metrics['acc']}")

# --- Step 3: Evaluate on held-out test set ---
print("=" * 60)
print("STEP 3: Evaluating on test set")
print("=" * 60)

cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    forward_batch_size=200,
    nproc=10,
)

# Find the saved model directory
import glob
model_dirs = glob.glob(f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/")
if model_dirs:
    saved_model_dir = model_dirs[0]
else:
    # Try to find it
    saved_model_dir = glob.glob(f"{output_dir}/*cellClassifier*/ksplit1/")[0]

print(f"Using saved model: {saved_model_dir}")

all_metrics_test = cc_eval.evaluate_saved_model(
    model_directory=saved_model_dir,
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

print(f"\nTest metrics:")
print(f"  Macro F1: {all_metrics_test['macro_f1']}")
print(f"  Accuracy: {all_metrics_test['acc']}")
print(f"  Confusion matrix:\n{all_metrics_test['conf_matrix']}")

# --- Step 4: Plot results ---
print("=" * 60)
print("STEP 4: Plotting results")
print("=" * 60)

cc_eval.plot_conf_mat(
    conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
    output_directory=output_dir,
    output_prefix=output_prefix,
    custom_class_order=["nf", "hcm", "dcm"],
)

cc_eval.plot_predictions(
    predictions_file=f"{output_dir}/{output_prefix}_pred_dict.pkl",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    title="disease",
    output_directory=output_dir,
    output_prefix=output_prefix,
    custom_class_order=["nf", "hcm", "dcm"],
)

print(f"\nAll outputs saved to: {output_dir}")
print("CELL CLASSIFICATION COMPLETE!")
