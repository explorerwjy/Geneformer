"""
04_finetune_gf_ephys.py

Fine-tune Geneformer V1-10M with a regression head for multi-target
electrophysiology prediction from Patch-seq transcriptomes.

Architecture:
    [tokenized cell] -> GF-V1 (6L, 256d BERT) -> BertPooler (CLS -> Linear(256,256) + Tanh)
                     -> Dropout -> Linear(256, N_targets) -> N ephys features

Usage:
    python 04_finetune_gf_ephys.py --dataset gaba --round 1 --cv 10
    python 04_finetune_gf_ephys.py --dataset excitatory --round 1 --cv 0
    python 04_finetune_gf_ephys.py --dataset pooled --round 1 --cv 10
"""

import argparse
import json
import logging
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    BertForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")
from geneformer import TOKEN_DICTIONARY_FILE_30M
from geneformer.collator_for_classification import DataCollatorForCellClassification

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(
    "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V1-10M"
)
DATA_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data")
RESULTS_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/results")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token dictionary (V1 uses the 30M corpus dictionary)
# ---------------------------------------------------------------------------
with open(TOKEN_DICTIONARY_FILE_30M, "rb") as f:
    TOKEN_DICT = pickle.load(f)


# ---------------------------------------------------------------------------
# Data collator for multi-target float regression
# ---------------------------------------------------------------------------
class DataCollatorForEphysRegression(DataCollatorForCellClassification):
    """
    Extends cell classification collator to handle float regression labels.
    The parent class handles padding of input_ids and attention_mask.
    We override __call__ to cast labels to float32 (for MSE loss) and
    everything else to int64.
    """

    def __call__(self, features):
        batch = self._prepare_batch(features)
        result = {}
        for k, v in batch.items():
            if k == "labels":
                # Regression targets must be float32
                result[k] = (
                    v.float()
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v, dtype=torch.float32)
                )
            else:
                result[k] = (
                    v.to(torch.int64)
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v, dtype=torch.int64)
                )
        return result


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------
def load_ephys(dataset: str, round_num: int) -> dict:
    """Load ephys pickle for the requested dataset/round."""
    if dataset == "gaba":
        if round_num == 1:
            path = DATA_DIR / "gaba_ephys_r1.pkl"
        else:
            path = DATA_DIR / "gaba_ephys_r2.pkl"
    elif dataset == "excitatory":
        path = DATA_DIR / "excitatory_ephys.pkl"
    elif dataset == "pooled":
        # Pooled uses GABA R1 + excitatory (both have the same 18 features)
        path = None  # handled below
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if dataset == "pooled":
        with open(DATA_DIR / "gaba_ephys_r1.pkl", "rb") as f:
            gaba = pickle.load(f)
        with open(DATA_DIR / "excitatory_ephys.pkl", "rb") as f:
            exc = pickle.load(f)
        # Concatenate ephys DataFrames
        ephys = pd.concat([gaba["ephys"], exc["ephys"]], axis=0)
        donor_map = {**gaba["donor_map"], **exc["donor_map"]}
        feature_names = gaba["feature_names"]
        return {
            "ephys": ephys,
            "donor_map": donor_map,
            "feature_names": feature_names,
        }
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_tokenized(dataset: str) -> Dataset:
    """Load tokenized HuggingFace dataset for the requested dataset."""
    if dataset == "gaba":
        return load_from_disk(str(DATA_DIR / "gaba_tokenized.dataset"))
    elif dataset == "excitatory":
        return load_from_disk(str(DATA_DIR / "excitatory_tokenized.dataset"))
    elif dataset == "pooled":
        gaba = load_from_disk(str(DATA_DIR / "gaba_tokenized.dataset"))
        exc = load_from_disk(str(DATA_DIR / "excitatory_tokenized.dataset"))
        return concatenate_datasets([gaba, exc])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def align_data(
    tokenized_ds: Dataset, ephys_data: dict
) -> tuple[Dataset, np.ndarray, list[str]]:
    """
    Align tokenized dataset with ephys labels by specimen_id.
    Drops rows with NaN ephys values (simpler than masking).

    Returns:
        aligned_ds: HF Dataset with 'label' column (list of floats per row)
        labels_array: np.ndarray of shape (n_cells, n_features)
        feature_names: list of feature names
    """
    ephys_df = ephys_data["ephys"]
    feature_names = ephys_data["feature_names"]

    # Drop NaN rows from ephys before alignment
    n_before = len(ephys_df)
    ephys_df = ephys_df.dropna()
    n_dropped = n_before - len(ephys_df)
    if n_dropped > 0:
        logger.info(
            f"Dropped {n_dropped} cells with NaN ephys values "
            f"({n_before} -> {len(ephys_df)})"
        )

    # Build lookup: specimen_id -> index in ephys_df
    ephys_ids = set(ephys_df.index)

    # Get specimen_ids from tokenized dataset
    tok_ids = tokenized_ds["specimen_id"]

    # Find intersection
    keep_idx = [i for i, sid in enumerate(tok_ids) if sid in ephys_ids]
    logger.info(
        f"Alignment: {len(tok_ids)} tokenized cells, "
        f"{len(ephys_df)} ephys cells, "
        f"{len(keep_idx)} matched"
    )

    if len(keep_idx) == 0:
        raise ValueError("No matching specimen_ids between tokenized data and ephys!")

    # Subset tokenized dataset
    aligned_ds = tokenized_ds.select(keep_idx)

    # Build labels array aligned with the subsetted dataset
    aligned_sids = [tok_ids[i] for i in keep_idx]
    labels_array = ephys_df.loc[aligned_sids][feature_names].values.astype(np.float32)

    # Add labels to dataset (as list of floats per row for the collator)
    label_lists = [row.tolist() for row in labels_array]
    aligned_ds = aligned_ds.add_column("label", label_lists)

    return aligned_ds, labels_array, feature_names


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def load_and_freeze_model(n_targets: int) -> BertForSequenceClassification:
    """
    Load GF-V1-10M as BertForSequenceClassification with regression head.
    Freeze embeddings + encoder layers 0-3, fine-tune layers 4-5 + classifier.
    """
    model = BertForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        num_labels=n_targets,
        problem_type="regression",
        output_hidden_states=False,
        output_attentions=False,
    )

    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze encoder layers 0-3 (keep 4-5 unfrozen)
    for layer_idx in range(4):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model loaded: {total_params:,} total params, "
        f"{trainable_params:,} trainable ({100*trainable_params/total_params:.1f}%)"
    )

    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def make_compute_metrics(n_features: int):
    """Return a compute_metrics function that computes R2 and mean Pearson r."""

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Global R2 (uniform average across targets)
        r2 = r2_score(labels, preds, multioutput="uniform_average")

        # Per-feature Pearson r
        pearson_rs = []
        for j in range(n_features):
            # Guard against constant columns
            if np.std(labels[:, j]) < 1e-10 or np.std(preds[:, j]) < 1e-10:
                pearson_rs.append(0.0)
            else:
                r, _ = pearsonr(labels[:, j], preds[:, j])
                pearson_rs.append(float(r) if np.isfinite(r) else 0.0)

        return {
            "r2": float(r2),
            "mean_pearson_r": float(np.mean(pearson_rs)),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Cross-validation splitters
# ---------------------------------------------------------------------------
def get_kfold_splits(
    n_samples: int, n_splits: int, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Standard K-fold CV splits."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(range(n_samples)))


def get_lodo_splits(
    specimen_ids: list[int], donor_map: dict
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Leave-one-donor-out CV: each fold holds out all cells from one donor.
    Only includes donors that have at least 1 cell in the aligned dataset.
    """
    # Map each specimen to its donor
    sid_to_donor = {}
    for sid in specimen_ids:
        donor = donor_map.get(sid) or donor_map.get(int(sid))
        if donor is not None:
            sid_to_donor[sid] = donor

    # Group indices by donor
    donor_to_indices = {}
    for i, sid in enumerate(specimen_ids):
        donor = sid_to_donor.get(sid)
        if donor is not None:
            donor_to_indices.setdefault(donor, []).append(i)

    n_missing = len(specimen_ids) - len(sid_to_donor)
    if n_missing > 0:
        logger.warning(
            f"{n_missing} cells have no donor mapping; they will be excluded "
            f"from LODO evaluation folds."
        )

    all_indices = set(range(len(specimen_ids)))
    splits = []
    for donor, test_idx in sorted(donor_to_indices.items()):
        test_idx = np.array(test_idx)
        train_idx = np.array(sorted(all_indices - set(test_idx)))
        if len(test_idx) > 0 and len(train_idx) > 0:
            splits.append((train_idx, test_idx))

    logger.info(f"LODO: {len(splits)} donors -> {len(splits)} folds")
    return splits


# ---------------------------------------------------------------------------
# Training loop (one fold)
# ---------------------------------------------------------------------------
def train_one_fold(
    fold_idx: int,
    train_ds: Dataset,
    eval_ds: Dataset,
    n_targets: int,
    output_dir: Path,
) -> dict:
    """Train a single fold and return evaluation metrics."""
    model = load_and_freeze_model(n_targets)

    training_args = TrainingArguments(
        output_dir=str(output_dir / f"fold_{fold_idx}"),
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=4,
        seed=42,
    )

    data_collator = DataCollatorForEphysRegression(
        token_dictionary=TOKEN_DICT,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(n_targets),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    # Evaluate on held-out set
    eval_results = trainer.evaluate()
    logger.info(f"Fold {fold_idx}: R2={eval_results['eval_r2']:.4f}, "
                f"mean_pearson_r={eval_results['eval_mean_pearson_r']:.4f}")

    # Get per-feature predictions for detailed metrics
    predictions = trainer.predict(eval_ds)
    preds = predictions.predictions
    labels = predictions.label_ids

    # Per-feature R2
    per_feature_r2 = {}
    for j in range(n_targets):
        try:
            per_feature_r2[f"feature_{j}"] = float(
                r2_score(labels[:, j], preds[:, j])
            )
        except Exception:
            per_feature_r2[f"feature_{j}"] = float("nan")

    fold_metrics = {
        "fold": fold_idx,
        "r2": float(eval_results["eval_r2"]),
        "mean_pearson_r": float(eval_results["eval_mean_pearson_r"]),
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "per_feature_r2": per_feature_r2,
    }

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return fold_metrics


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    dataset: str, round_num: int, cv: int
) -> dict:
    """Run the full fine-tuning experiment with cross-validation."""

    logger.info(f"Dataset: {dataset}, Round: {round_num}, CV: {cv}")

    # Load data
    ephys_data = load_ephys(dataset, round_num)
    tokenized_ds = load_tokenized(dataset)
    aligned_ds, labels_array, feature_names = align_data(tokenized_ds, ephys_data)
    n_targets = len(feature_names)
    n_samples = len(aligned_ds)

    logger.info(f"Aligned: {n_samples} cells, {n_targets} targets")
    logger.info(f"Features: {feature_names}")

    # Standardize labels (z-score per feature) for training stability
    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(labels_array)

    # Replace the label column with scaled values
    label_lists_scaled = [row.tolist() for row in labels_scaled]
    aligned_ds = aligned_ds.remove_columns("label")
    aligned_ds = aligned_ds.add_column("label", label_lists_scaled)

    # Get specimen_ids for LODO splits
    specimen_ids = aligned_ds["specimen_id"]

    # Determine splits
    if cv > 0:
        splits = get_kfold_splits(n_samples, n_splits=cv)
        cv_name = f"cv{cv}"
    else:
        splits = get_lodo_splits(specimen_ids, ephys_data["donor_map"])
        cv_name = "lodo"

    logger.info(f"Running {len(splits)} folds ({cv_name})")

    # Output directory
    out_dir = RESULTS_DIR / dataset / f"round{round_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run folds
    all_fold_metrics = []

    for fold_idx, (train_idx, eval_idx) in enumerate(splits):
        logger.info(
            f"\n{'='*60}\n"
            f"Fold {fold_idx + 1}/{len(splits)}: "
            f"{len(train_idx)} train, {len(eval_idx)} eval\n"
            f"{'='*60}"
        )

        train_ds = aligned_ds.select(train_idx.tolist())
        eval_ds = aligned_ds.select(eval_idx.tolist())

        # Use a temporary directory for model checkpoints to avoid clutter
        with tempfile.TemporaryDirectory() as tmp_dir:
            fold_metrics = train_one_fold(
                fold_idx=fold_idx,
                train_ds=train_ds,
                eval_ds=eval_ds,
                n_targets=n_targets,
                output_dir=Path(tmp_dir),
            )

        all_fold_metrics.append(fold_metrics)

    # Aggregate results
    fold_r2s = [m["r2"] for m in all_fold_metrics]

    # Per-feature R2 aggregated across folds
    per_feature_r2_agg = {}
    for j in range(n_targets):
        feat_key = f"feature_{j}"
        vals = [m["per_feature_r2"][feat_key] for m in all_fold_metrics
                if np.isfinite(m["per_feature_r2"][feat_key])]
        per_feature_r2_agg[feature_names[j]] = {
            "mean_r2": float(np.mean(vals)) if vals else float("nan"),
            "std_r2": float(np.std(vals)) if vals else float("nan"),
        }

    results = {
        "dataset": dataset,
        "round": round_num,
        "cv_type": cv_name,
        "n_folds": len(splits),
        "n_cells": n_samples,
        "n_features": n_targets,
        "feature_names": feature_names,
        "mean_fold_r2": float(np.mean(fold_r2s)),
        "std_fold_r2": float(np.std(fold_r2s)),
        "per_feature_r2": per_feature_r2_agg,
        "fold_metrics": all_fold_metrics,
    }

    # Save results
    results_file = out_dir / f"{cv_name}_gf_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(
        f"Mean fold R2: {results['mean_fold_r2']:.4f} "
        f"+/- {results['std_fold_r2']:.4f}"
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Geneformer V1-10M for ephys regression"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gaba", "excitatory", "pooled"],
        help="Dataset to use: gaba, excitatory, or pooled (gaba+excitatory)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        choices=[1, 2],
        help="Feature round: 1=18 shared features, 2=80 GABA features (default: 1)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=10,
        help="CV folds: >0 for K-fold, 0 for leave-one-donor-out (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate: excitatory and pooled only support round 1
    if args.dataset in ("excitatory", "pooled") and args.round == 2:
        logger.error(
            f"Round 2 (80 GABA features) is only available for --dataset gaba. "
            f"Got --dataset {args.dataset} --round {args.round}."
        )
        sys.exit(1)

    results = run_experiment(
        dataset=args.dataset,
        round_num=args.round,
        cv=args.cv,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
