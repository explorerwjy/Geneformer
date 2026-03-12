"""Phase 2B: Evaluate the already-trained classifier on held-out test set."""
import matplotlib
matplotlib.use("Agg")
import glob
import os
from geneformer import Classifier

output_dir = "outputs/disease_classification/260303213710"
output_prefix = "disease_classifier"
saved_model_dir = glob.glob(f"{output_dir}/*cellClassifier*/ksplit1/")[0]
print(f"Using saved model: {saved_model_dir}")

cc_eval = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    forward_batch_size=25,
    model_version="V2",
    nproc=10,
    ngpu=1,
)

all_metrics_test = cc_eval.evaluate_saved_model(
    model_directory=saved_model_dir,
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
    output_directory=output_dir,
    output_prefix=output_prefix,
)
macro_f1 = all_metrics_test["macro_f1"]
acc = all_metrics_test["acc"]
conf = all_metrics_test["conf_matrix"]
print(f"Test Macro F1: {macro_f1:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Confusion matrix:\n{conf}")

cc_eval.plot_conf_mat(
    conf_mat_dict={"Geneformer-V2-316M": conf},
    output_directory=output_dir,
    output_prefix=output_prefix,
    custom_class_order=["NF", "HCM", "DCM"],
)
cc_eval.plot_predictions(
    predictions_file=f"{output_dir}/{output_prefix}_pred_dict.pkl",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    title="Cardiomyopathy Disease Classification (V2-316M)",
    output_directory=output_dir,
    output_prefix=output_prefix,
    custom_class_order=["NF", "HCM", "DCM"],
)
print("Phase 2B evaluation complete!")
