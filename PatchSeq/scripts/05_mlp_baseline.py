#!/usr/bin/env python
"""
05_mlp_baseline.py

MLP baseline using human ion channel + marker gene expression to predict
electrophysiology features. Same evaluation protocol as GF fine-tuning
(z-scored targets, K-fold or LODO CV) for direct comparison.

Architecture:
    [N_ic genes, log2(CPM+1)] -> Linear(N_ic, 256) -> Tanh -> Dropout(0.1) -> Linear(256, N_targets)

Usage:
    python 05_mlp_baseline.py --dataset {gaba,excitatory,pooled} --round {1,2} --cv {10,0}
    # --cv 0 means leave-one-donor-out (LODO)
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATCHSEQ_DIR = PROJECT_ROOT / "PatchSeq"
DATA_DIR = PATCHSEQ_DIR / "data"
RESULTS_DIR = PATCHSEQ_DIR / "results"

# ---------------------------------------------------------------------------
# Ion channel gene prefixes and marker genes
# ---------------------------------------------------------------------------
IC_PREFIXES = [
    "SCN1", "SCN2", "SCN3", "SCN4", "SCN5", "SCN7", "SCN8", "SCN9", "SCN10", "SCN11",
    "KCNA", "KCNB", "KCNC", "KCND", "KCNE", "KCNF", "KCNG", "KCNH", "KCNJ",
    "KCNK", "KCNMA", "KCNMB", "KCNN", "KCNQ", "KCNS", "KCNT", "KCNU", "KCNV",
    "CACNA", "CACNB", "CACNG",
    "HCN",
    "CLCN",
]
MARKER_GENES = ["PVALB", "SST", "VIP", "LAMP5"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLPRegressor(nn.Module):
    """Simple 1-hidden-layer MLP for multi-output regression."""

    def __init__(self, n_input: int, n_output: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_output),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def select_ic_genes(adata: ad.AnnData) -> list[str]:
    """Select ion channel + marker genes present in the h5ad var."""
    symbols = adata.var["gene_symbol"].tolist()
    symbol_set = set(symbols)

    ic_genes = sorted({s for s in symbols if any(s.startswith(p) for p in IC_PREFIXES)})
    markers = sorted([g for g in MARKER_GENES if g in symbol_set])
    selected = sorted(set(ic_genes + markers))
    return selected


def load_expression(dataset: str) -> tuple[pd.DataFrame, ad.AnnData]:
    """
    Load h5ad and return log2(CPM+1) expression for IC+marker genes,
    indexed by specimen_id.

    Returns
    -------
    expr_df : DataFrame (n_cells x n_genes), log2(CPM+1) values
    adata   : the full AnnData (for reference)
    """
    if dataset == "pooled":
        adata_gaba = ad.read_h5ad(DATA_DIR / "gaba_raw_counts.h5ad")
        adata_exc = ad.read_h5ad(DATA_DIR / "excitatory_raw_counts.h5ad")
        # ad.concat drops var columns; recover gene_symbol from source
        var_ref = adata_gaba.var[["gene_symbol"]].copy()
        adata = ad.concat([adata_gaba, adata_exc], join="inner")
        adata.var = adata.var.join(var_ref)
    else:
        adata = ad.read_h5ad(DATA_DIR / f"{dataset}_raw_counts.h5ad")

    # Select IC + marker genes
    selected_genes = select_ic_genes(adata)
    gene_mask = adata.var["gene_symbol"].isin(selected_genes)
    adata_sub = adata[:, gene_mask].copy()

    # .X is CPM (sparse). Convert to dense, then log2(CPM+1).
    X_cpm = np.asarray(adata_sub.X.todense())
    X_log2 = np.log2(X_cpm + 1.0)

    expr_df = pd.DataFrame(
        X_log2,
        index=adata.obs["specimen_id"].values,
        columns=adata_sub.var["gene_symbol"].values,
    )
    expr_df.index.name = "specimen_id"

    return expr_df, adata


def load_ephys(dataset: str, round_num: int) -> tuple[pd.DataFrame, dict]:
    """
    Load ephys DataFrame and donor_map from pickle.

    Returns
    -------
    ephys : DataFrame indexed by specimen_id
    donor_map : dict {specimen_id: donor_id}
    """
    if dataset == "gaba":
        fname = f"gaba_ephys_r{round_num}.pkl"
    elif dataset == "excitatory":
        fname = "excitatory_ephys.pkl"
    elif dataset == "pooled":
        # Pool GABA (round-specific) + excitatory
        gaba_path = DATA_DIR / f"gaba_ephys_r{round_num}.pkl"
        exc_path = DATA_DIR / "excitatory_ephys.pkl"

        with open(gaba_path, "rb") as f:
            gaba_data = pickle.load(f)
        with open(exc_path, "rb") as f:
            exc_data = pickle.load(f)

        # Find shared features
        shared_cols = sorted(set(gaba_data["feature_names"]) & set(exc_data["feature_names"]))
        if not shared_cols:
            raise ValueError("No shared ephys features between GABA and excitatory datasets")

        ephys = pd.concat(
            [gaba_data["ephys"][shared_cols], exc_data["ephys"][shared_cols]],
            axis=0,
        )
        donor_map = {**gaba_data["donor_map"], **exc_data["donor_map"]}
        return ephys, donor_map
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    path = DATA_DIR / fname
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["ephys"], data["donor_map"]


def align_data(
    expr_df: pd.DataFrame,
    ephys: pd.DataFrame,
    donor_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Align expression and ephys by specimen_id, drop rows with NaN in ephys.

    Returns aligned (expr, ephys, donor_map) with matching indices.
    """
    shared_ids = sorted(set(expr_df.index) & set(ephys.index))
    if not shared_ids:
        raise ValueError("No shared specimen_ids between expression and ephys data")

    expr_aligned = expr_df.loc[shared_ids]
    ephys_aligned = ephys.loc[shared_ids]

    # Drop rows with any NaN in ephys
    valid_mask = ~ephys_aligned.isna().any(axis=1)
    expr_aligned = expr_aligned.loc[valid_mask]
    ephys_aligned = ephys_aligned.loc[valid_mask]

    donor_map_aligned = {
        sid: donor_map[sid] for sid in ephys_aligned.index if sid in donor_map
    }

    return expr_aligned, ephys_aligned, donor_map_aligned


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = "cuda",
) -> tuple[np.ndarray, float]:
    """
    Train MLP with early stopping on test R^2.

    Returns
    -------
    y_pred : predictions on test set (n_test x n_targets)
    best_r2 : best test R^2 achieved
    """
    n_input = X_train.shape[1]
    n_output = y_train.shape[1]

    model = MLPRegressor(n_input, n_output).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    best_r2 = -np.inf
    best_preds = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train_t)
        loss = criterion(pred_train, y_train_t)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
            pred_np = pred_test.cpu().numpy()
            test_r2 = r2_score(y_test, pred_np, multioutput="uniform_average")

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_preds = pred_np.copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_preds, best_r2


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Compute R^2 and per-feature Pearson r."""
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

    per_feature = {}
    for i, fname in enumerate(feature_names):
        r, p = pearsonr(y_true[:, i], y_pred[:, i])
        per_feature[fname] = {"pearson_r": float(r), "pearson_p": float(p)}

    return {
        "r2": float(r2),
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def run_kfold_cv(
    expr: pd.DataFrame,
    ephys: pd.DataFrame,
    n_splits: int = 10,
    device: str = "cuda",
) -> dict:
    """K-fold cross-validation."""
    X = expr.values
    y = ephys.values
    feature_names = list(ephys.columns)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_preds = np.zeros_like(y)
    all_true = np.zeros_like(y)
    all_idx = np.zeros(len(y), dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Z-score expression on train
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_train_z = (X_train - X_mean) / X_std
        X_test_z = (X_test - X_mean) / X_std

        # Z-score targets on train
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0)
        y_std[y_std == 0] = 1.0
        y_train_z = (y_train - y_mean) / y_std
        y_test_z = (y_test - y_mean) / y_std

        preds_z, best_r2 = train_mlp(X_train_z, y_train_z, X_test_z, y_test_z, device=device)

        # Un-z-score predictions for computing metrics in original space
        preds_orig = preds_z * y_std + y_mean

        metrics = compute_metrics(y_test, preds_orig, feature_names)
        fold_results.append(metrics)

        all_preds[test_idx] = preds_orig
        all_true[test_idx] = y_test
        all_idx[test_idx] = np.arange(len(test_idx))

        print(f"  Fold {fold_i+1}/{n_splits}: R2={metrics['r2']:.4f}")

    # Overall metrics on pooled OOS predictions
    overall = compute_metrics(all_true, all_preds, feature_names)

    return {
        "cv_type": f"{n_splits}-fold",
        "n_samples": len(X),
        "n_features": len(feature_names),
        "n_genes": X.shape[1],
        "overall_r2": overall["r2"],
        "overall_per_feature": overall["per_feature"],
        "fold_r2s": [fr["r2"] for fr in fold_results],
        "mean_fold_r2": float(np.mean([fr["r2"] for fr in fold_results])),
        "std_fold_r2": float(np.std([fr["r2"] for fr in fold_results])),
        "feature_names": feature_names,
    }


def run_lodo_cv(
    expr: pd.DataFrame,
    ephys: pd.DataFrame,
    donor_map: dict,
    device: str = "cuda",
) -> dict:
    """Leave-one-donor-out cross-validation."""
    X = expr.values
    y = ephys.values
    specimen_ids = expr.index.values
    feature_names = list(ephys.columns)

    # Map specimen_id -> donor
    donors = np.array([donor_map[sid] for sid in specimen_ids])
    unique_donors = sorted(set(donors))
    n_donors = len(unique_donors)

    fold_results = []
    all_preds = np.full_like(y, np.nan)

    for di, donor in enumerate(unique_donors):
        test_mask = donors == donor
        train_mask = ~test_mask

        n_test = test_mask.sum()
        if n_test == 0:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Z-score expression on train
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_train_z = (X_train - X_mean) / X_std
        X_test_z = (X_test - X_mean) / X_std

        # Z-score targets on train
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0)
        y_std[y_std == 0] = 1.0
        y_train_z = (y_train - y_mean) / y_std
        y_test_z = (y_test - y_mean) / y_std

        preds_z, best_r2 = train_mlp(X_train_z, y_train_z, X_test_z, y_test_z, device=device)

        preds_orig = preds_z * y_std + y_mean

        metrics = compute_metrics(y_test, preds_orig, feature_names)
        fold_results.append({"donor": donor, "n_cells": int(n_test), **metrics})

        all_preds[test_mask] = preds_orig

        if (di + 1) % 10 == 0 or (di + 1) == n_donors:
            print(f"  LODO {di+1}/{n_donors}: donor={donor}, n={n_test}, R2={metrics['r2']:.4f}")

    # Overall metrics on pooled OOS predictions
    valid = ~np.isnan(all_preds[:, 0])
    overall = compute_metrics(y[valid], all_preds[valid], feature_names)

    return {
        "cv_type": "LODO",
        "n_samples": int(valid.sum()),
        "n_donors": n_donors,
        "n_features": len(feature_names),
        "n_genes": X.shape[1],
        "overall_r2": overall["r2"],
        "overall_per_feature": overall["per_feature"],
        "fold_r2s": [fr["r2"] for fr in fold_results],
        "mean_fold_r2": float(np.mean([fr["r2"] for fr in fold_results])),
        "std_fold_r2": float(np.std([fr["r2"] for fr in fold_results])),
        "per_donor": fold_results,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLP baseline on IC gene expression")
    parser.add_argument(
        "--dataset",
        choices=["gaba", "excitatory", "pooled"],
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--round",
        type=int,
        choices=[1, 2],
        default=1,
        help="Ephys feature round (1=18 shared, 2=80 GABA-expanded). Only affects GABA.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=10,
        help="Number of CV folds (0=LODO)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print(f"\nLoading expression data ({args.dataset})...")
    expr_df, adata = load_expression(args.dataset)
    print(f"  Expression: {expr_df.shape[0]} cells x {expr_df.shape[1]} IC+marker genes")
    print(f"  Gene names: {list(expr_df.columns[:5])} ... ({expr_df.shape[1]} total)")

    print(f"\nLoading ephys data ({args.dataset}, round {args.round})...")
    ephys, donor_map = load_ephys(args.dataset, args.round)
    print(f"  Ephys: {ephys.shape[0]} cells x {ephys.shape[1]} features")

    print("\nAligning data...")
    expr_aligned, ephys_aligned, donor_map_aligned = align_data(expr_df, ephys, donor_map)
    print(f"  Aligned: {expr_aligned.shape[0]} cells")
    print(f"  Ephys features: {list(ephys_aligned.columns)}")

    # -----------------------------------------------------------------------
    # Run CV
    # -----------------------------------------------------------------------
    t0 = time.time()
    if args.cv == 0:
        print(f"\nRunning LODO CV ({len(set(donor_map_aligned.values()))} donors)...")
        results = run_lodo_cv(expr_aligned, ephys_aligned, donor_map_aligned, device=device)
        cv_label = "lodo"
    else:
        print(f"\nRunning {args.cv}-fold CV...")
        results = run_kfold_cv(expr_aligned, ephys_aligned, n_splits=args.cv, device=device)
        cv_label = f"cv{args.cv}"
    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Results ({args.dataset}, round {args.round}, {cv_label})")
    print(f"{'='*60}")
    print(f"  Overall R2: {results['overall_r2']:.4f}")
    print(f"  Mean fold R2: {results['mean_fold_r2']:.4f} +/- {results['std_fold_r2']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Per-feature Pearson r summary
    print("\n  Per-feature Pearson r:")
    for fname, fdata in sorted(results["overall_per_feature"].items()):
        print(f"    {fname:45s} r={fdata['pearson_r']:.4f}  p={fdata['pearson_p']:.2e}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_dir = RESULTS_DIR / args.dataset / f"round{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cv_label}_mlp_results.json"

    # Add metadata
    results["dataset"] = args.dataset
    results["round"] = args.round
    results["model"] = "MLP"
    results["architecture"] = f"Linear({results['n_genes']}, 256) -> Tanh -> Dropout(0.1) -> Linear(256, {results['n_features']})"
    results["elapsed_seconds"] = float(elapsed)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
