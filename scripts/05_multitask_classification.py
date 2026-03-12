"""
Multi-Task Cell Classification: cell_type + disease on cardiomyopathy data
Based on examples/multitask_cell_classification.ipynb
Uses V1 pretrained model with disease dataset (has both cell_type and disease columns).
"""
import matplotlib
matplotlib.use("Agg")
import datetime
import os
import numpy as np
from datasets import load_from_disk
from geneformer import MTLClassifier

# Paths
MODEL_DIR = "/home/jw3514/Work/Geneformer/models/Geneformer/Geneformer-V1-10M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"

output_dir = f"/home/jw3514/Work/Geneformer/outputs/multitask/{datestamp}"
os.makedirs(output_dir, exist_ok=True)

# --- Step 1: Prepare train/val/test splits ---
print("=" * 60)
print("STEP 1: Preparing train/val/test splits")
print("=" * 60)

ds = load_from_disk(INPUT_DATA)

# Add unique_cell_id column (required by MTLClassifier)
ds = ds.map(lambda example, idx: {"unique_cell_id": f"cell_{idx}"}, with_indices=True, num_proc=10)

# Filter to cardiomyocytes for a focused task
ds = ds.filter(
    lambda x: x["cell_type"] in ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"],
    num_proc=10,
)
print(f"Filtered dataset: {len(ds)} cardiomyocytes")

# Split by individual (same as cell classification notebook)
train_ids = ["1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561",
             "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617",
             "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371",
             "1549", "1515"]
eval_ids = ["1422", "1510", "1539", "1606", "1702"]
test_ids = ["1437", "1516", "1602", "1685", "1718"]

train_ds = ds.filter(lambda x: x["individual"] in train_ids, num_proc=10)
val_ds = ds.filter(lambda x: x["individual"] in eval_ids, num_proc=10)
test_ds = ds.filter(lambda x: x["individual"] in test_ids, num_proc=10)

# Subsample for faster runtime
max_train = 20000
max_val = 5000
max_test = 5000

if len(train_ds) > max_train:
    np.random.seed(42)
    indices = np.random.choice(len(train_ds), max_train, replace=False)
    train_ds = train_ds.select(indices)
if len(val_ds) > max_val:
    np.random.seed(42)
    indices = np.random.choice(len(val_ds), max_val, replace=False)
    val_ds = val_ds.select(indices)
if len(test_ds) > max_test:
    np.random.seed(42)
    indices = np.random.choice(len(test_ds), max_test, replace=False)
    test_ds = test_ds.select(indices)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Save splits
train_path = f"{output_dir}/train.dataset"
val_path = f"{output_dir}/val.dataset"
test_path = f"{output_dir}/test.dataset"
train_ds.save_to_disk(train_path)
val_ds.save_to_disk(val_path)
test_ds.save_to_disk(test_path)
print("Splits saved!")

# --- Step 2: Run MTL with manual hyperparameters ---
print("=" * 60)
print("STEP 2: Running multi-task classification (cell_type + disease)")
print("=" * 60)

task_columns = ["cell_type", "disease"]

manual_hyperparameters = {
    "learning_rate": 0.001,
    "warmup_ratio": 0.01,
    "weight_decay": 0.1,
    "dropout_rate": 0.1,
    "lr_scheduler_type": "cosine",
    "task_weights": [1, 1],
    "max_layers_to_freeze": 2,
}

mc = MTLClassifier(
    task_columns=task_columns,
    study_name="mtl_cardiomyopathy",
    pretrained_path=MODEL_DIR,
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    model_save_path=f"{output_dir}/model",
    results_dir=output_dir,
    tensorboard_log_dir=f"{output_dir}/tensorboard",
    manual_hyperparameters=manual_hyperparameters,
    use_manual_hyperparameters=True,
    epochs=1,
    batch_size=12,
    seed=42,
)

mc.run_manual_tuning()
print("Training complete!")

# --- Step 3: Evaluate on test data ---
print("=" * 60)
print("STEP 3: Evaluating on test set")
print("=" * 60)

mc.load_and_evaluate_test_model()

print(f"\nAll outputs saved to: {output_dir}")
print("MULTITASK CLASSIFICATION COMPLETE!")
