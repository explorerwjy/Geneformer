"""
Gene Classification: Dosage-Sensitive vs -Insensitive Transcription Factors
Based on examples/gene_classification.ipynb
Uses V1 model with dosage sensitivity dataset.
"""
import datetime
import os
import pickle
from geneformer import Classifier

# Paths
MODEL_DIR = "/home/jw3514/Work/Geneformer/models/Geneformer/Geneformer-V1-10M"
INPUT_DATA = "/home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/gene_classification/dosage_sensitive_tfs/gc-30M_sample50k.dataset"
GENE_CLASS_DICT = "/home/jw3514/Work/Geneformer/data/Genecorpus-30M/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sensitivity_TFs.pickle"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "tf_dosage_sens"
output_dir = f"/home/jw3514/Work/Geneformer/outputs/gene_classification/{datestamp}"
os.makedirs(output_dir, exist_ok=True)

# --- Step 1: Load gene class dictionary ---
print("=" * 60)
print("STEP 1: Loading gene class dictionary")
print("=" * 60)

with open(GENE_CLASS_DICT, "rb") as fp:
    gene_class_dict = pickle.load(fp)

print(f"Gene classes: {list(gene_class_dict.keys())}")
for k, v in gene_class_dict.items():
    print(f"  {k}: {len(v)} genes")

# --- Step 2: Train with 5-fold cross-validation ---
print("=" * 60)
print("STEP 2: Setting up 5-fold cross-validation")
print("=" * 60)

cc = Classifier(
    classifier="gene",
    gene_class_dict=gene_class_dict,
    max_ncells=10_000,
    freeze_layers=4,
    num_crossval_splits=5,
    forward_batch_size=200,
    model_version="V1",
    nproc=10,
)

cc.prepare_data(
    input_data_file=INPUT_DATA,
    output_directory=output_dir,
    output_prefix=output_prefix,
)
print("Data prepared!")

print("=" * 60)
print("STEP 3: Running 5-fold cross-validation")
print("=" * 60)

all_metrics = cc.validate(
    model_directory=MODEL_DIR,
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

print(f"\nCross-validation metrics:")
print(f"  Macro F1 per fold: {all_metrics['macro_f1']}")
print(f"  Mean Macro F1: {sum(all_metrics['macro_f1'])/len(all_metrics['macro_f1']):.4f}")
print(f"  Accuracy per fold: {all_metrics['acc']}")
print(f"  Mean Accuracy: {sum(all_metrics['acc'])/len(all_metrics['acc']):.4f}")
print(f"  ROC AUC: {all_metrics['all_roc_metrics']['roc_auc']:.4f} +/- {all_metrics['all_roc_metrics']['roc_auc_sd']:.4f}")

# --- Step 4: Plot results ---
print("=" * 60)
print("STEP 4: Plotting results")
print("=" * 60)

cc.plot_conf_mat(
    conf_mat_dict={"Geneformer": all_metrics["conf_matrix"]},
    output_directory=output_dir,
    output_prefix=output_prefix,
)

cc.plot_roc(
    roc_metric_dict={"Geneformer": all_metrics["all_roc_metrics"]},
    model_style_dict={"Geneformer": {"color": "red", "linestyle": "-"}},
    title="Dosage-sensitive vs -insensitive TFs",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

print(f"\nAll outputs saved to: {output_dir}")

# --- Step 5: Train on all data (produces final model) ---
print("=" * 60)
print("STEP 5: Training on all data")
print("=" * 60)

cc_all = Classifier(
    classifier="gene",
    gene_class_dict=gene_class_dict,
    max_ncells=10_000,
    freeze_layers=4,
    num_crossval_splits=0,
    forward_batch_size=200,
    model_version="V1",
    nproc=10,
)

cc_all.prepare_data(
    input_data_file=INPUT_DATA,
    output_directory=output_dir,
    output_prefix=f"{output_prefix}_alldata",
)

cc_all.train_all_data(
    model_directory=MODEL_DIR,
    prepared_input_data_file=f"{output_dir}/{output_prefix}_alldata_labeled.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_alldata_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=f"{output_prefix}_alldata",
)

print(f"\nAll outputs saved to: {output_dir}")
print("GENE CLASSIFICATION COMPLETE!")
