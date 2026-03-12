#!/usr/bin/env python
"""
06_analyze_results.py

Load all JSON result files from GF and MLP experiments, generate comparison
tables (printed to stdout) and publication-quality plots.

Outputs:
    PatchSeq/results/gf_vs_mlp_comparison.png   -- bar chart, GF vs MLP by dataset/CV
    PatchSeq/results/gaba_per_feature_r2.png     -- per-feature R2 for GABA R1 (GF)
    PatchSeq/results/lodo_comparison.png         -- LODO donor-level boxplot
    Cross-species table (stdout)                 -- human vs hardcoded mouse baselines

Usage:
    python 06_analyze_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATCHSEQ_DIR = PROJECT_ROOT / "PatchSeq"
RESULTS_DIR = PATCHSEQ_DIR / "results"

# ---------------------------------------------------------------------------
# Datasets and experiments to scan
# ---------------------------------------------------------------------------
DATASETS = ["gaba", "excitatory", "pooled"]
ROUNDS = {
    "gaba": [1, 2],
    "excitatory": [1],
    "pooled": [1],
}
CV_TYPES = ["cv10", "lodo"]
MODEL_TYPES = ["gf", "mlp"]

# Hardcoded mouse baselines from literature (for cross-species comparison)
MOUSE_BASELINES = {
    "M1": {"GF": 0.435, "MLP": 0.365},
    "V1": {"GF": 0.345, "MLP": 0.295},
}


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------
def load_result(dataset: str, round_num: int, cv_type: str, model: str) -> dict | None:
    """
    Attempt to load a result JSON file. Returns None if the file doesn't exist.

    Handles both the GF and MLP JSON schemas produced by scripts 04 and 05.
    """
    fname = f"{cv_type}_{model}_results.json"
    path = RESULTS_DIR / dataset / f"round{round_num}" / fname
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # Normalise: ensure 'model' key is present
        if "model" not in data:
            data["model"] = model.upper()
        return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: could not read {path}: {exc}")
        return None


def extract_mean_r2(result: dict) -> float | None:
    """Extract the mean fold R2 from a result dict, handling both schemas."""
    for key in ("mean_fold_r2", "global_r2"):
        if key in result and result[key] is not None:
            return float(result[key])
    return None


def extract_std_r2(result: dict) -> float | None:
    """Extract the std of fold R2 values."""
    if "std_fold_r2" in result and result["std_fold_r2"] is not None:
        return float(result["std_fold_r2"])
    return None


def extract_fold_r2s(result: dict) -> list[float] | None:
    """Extract per-fold R2 values for boxplot-style visualisations."""
    # MLP script stores fold_r2s; GF script stores fold_metrics[].r2
    if "fold_r2s" in result:
        return [float(v) for v in result["fold_r2s"]]
    if "fold_metrics" in result:
        return [float(fm["r2"]) for fm in result["fold_metrics"] if "r2" in fm]
    return None


def extract_per_feature_r2(result: dict) -> dict[str, float] | None:
    """
    Extract per-feature R2 values. The GF script stores these in
    per_feature_r2 as {feature_name: {mean_r2, std_r2}} while the MLP
    script stores overall_per_feature as {feature_name: {pearson_r, ...}}.
    We return {feature_name: float_value}.
    """
    if "per_feature_r2" in result:
        pf = result["per_feature_r2"]
        out = {}
        for k, v in pf.items():
            if isinstance(v, dict):
                out[k] = float(v.get("mean_r2", v.get("r2", 0.0)))
            else:
                out[k] = float(v)
        return out
    if "overall_per_feature" in result:
        # MLP stores Pearson r, not R2.  We'll note this in labels.
        pf = result["overall_per_feature"]
        return {k: float(v.get("pearson_r", 0.0)) for k, v in pf.items()}
    return None


# ---------------------------------------------------------------------------
# Scanning all results
# ---------------------------------------------------------------------------
def scan_all_results() -> list[dict]:
    """Scan for all expected result files and return a list of summary rows."""
    rows = []
    missing = []

    for dataset in DATASETS:
        for round_num in ROUNDS[dataset]:
            for cv_type in CV_TYPES:
                for model in MODEL_TYPES:
                    result = load_result(dataset, round_num, cv_type, model)
                    if result is None:
                        missing.append(
                            f"  {dataset}/round{round_num}/{cv_type}_{model}_results.json"
                        )
                        continue
                    mean_r2 = extract_mean_r2(result)
                    std_r2 = extract_std_r2(result)
                    rows.append(
                        {
                            "dataset": dataset,
                            "round": round_num,
                            "cv_type": cv_type,
                            "model": model.upper(),
                            "mean_r2": mean_r2,
                            "std_r2": std_r2,
                            "n_cells": result.get("n_cells", result.get("n_samples")),
                            "n_features": result.get("n_features"),
                        }
                    )

    return rows, missing


# ---------------------------------------------------------------------------
# 1) Summary table
# ---------------------------------------------------------------------------
def print_summary_table(rows: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    if not rows:
        print("\nNo result files found -- nothing to summarise.\n")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Dataset x Model x CV x R2")
    print("=" * 80)

    # Pivot for readability
    df["label"] = df["dataset"] + " R" + df["round"].astype(str)
    df["r2_str"] = df.apply(
        lambda r: (
            f"{r['mean_r2']:.4f} +/- {r['std_r2']:.4f}"
            if r["mean_r2"] is not None and r["std_r2"] is not None
            else "N/A"
        ),
        axis=1,
    )

    pivot = df.pivot_table(
        index=["label", "cv_type"],
        columns="model",
        values="r2_str",
        aggfunc="first",
    )
    # pd.set_option to avoid truncation
    with pd.option_context("display.max_colwidth", 40, "display.width", 120):
        print(pivot.to_string())

    print()

    # Also print a compact numeric version
    print("-" * 80)
    print(f"{'Dataset':20s} {'CV':8s} {'Model':6s} {'Mean R2':>10s} {'Std R2':>10s} {'N_cells':>8s}")
    print("-" * 80)
    for _, r in df.iterrows():
        mr2 = f"{r['mean_r2']:.4f}" if r["mean_r2"] is not None else "N/A"
        sr2 = f"{r['std_r2']:.4f}" if r["std_r2"] is not None else "N/A"
        nc = str(r["n_cells"]) if r["n_cells"] is not None else "?"
        print(f"{r['label']:20s} {r['cv_type']:8s} {r['model']:6s} {mr2:>10s} {sr2:>10s} {nc:>8s}")
    print("-" * 80)
    print()


# ---------------------------------------------------------------------------
# 2) GF vs MLP bar chart
# ---------------------------------------------------------------------------
def plot_gf_vs_mlp(rows: list[dict]) -> None:
    """Bar chart comparing GF and MLP mean R2 across dataset/cv combos."""
    if not rows:
        print("Skipping GF vs MLP plot -- no results.\n")
        return

    df = pd.DataFrame(rows)
    df["label"] = df["dataset"] + " R" + df["round"].astype(str) + " " + df["cv_type"]

    # We need both GF and MLP for at least one condition to make a useful plot
    labels = sorted(df["label"].unique())
    gf_vals = []
    gf_errs = []
    mlp_vals = []
    mlp_errs = []
    usable_labels = []

    for lab in labels:
        sub = df[df["label"] == lab]
        gf_row = sub[sub["model"] == "GF"]
        mlp_row = sub[sub["model"] == "MLP"]
        gf_r2 = gf_row["mean_r2"].values[0] if len(gf_row) else None
        mlp_r2 = mlp_row["mean_r2"].values[0] if len(mlp_row) else None
        gf_std = gf_row["std_r2"].values[0] if len(gf_row) else 0
        mlp_std = mlp_row["std_r2"].values[0] if len(mlp_row) else 0

        if gf_r2 is not None or mlp_r2 is not None:
            usable_labels.append(lab)
            gf_vals.append(gf_r2 if gf_r2 is not None else 0)
            gf_errs.append(gf_std if gf_std is not None else 0)
            mlp_vals.append(mlp_r2 if mlp_r2 is not None else 0)
            mlp_errs.append(mlp_std if mlp_std is not None else 0)

    if not usable_labels:
        print("Skipping GF vs MLP plot -- no usable conditions.\n")
        return

    x = np.arange(len(usable_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(usable_labels) * 1.5), 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    bars_gf = ax.bar(
        x - width / 2, gf_vals, width, yerr=gf_errs, label="Geneformer",
        color="#4C72B0", capsize=4, edgecolor="black", linewidth=0.5,
    )
    bars_mlp = ax.bar(
        x + width / 2, mlp_vals, width, yerr=mlp_errs, label="MLP (IC genes)",
        color="#DD8452", capsize=4, edgecolor="black", linewidth=0.5,
    )

    ax.set_ylabel("Mean R$^2$ (across folds)", fontsize=12)
    ax.set_title("Geneformer vs MLP Baseline: Ephys Prediction", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(usable_labels, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bars in [bars_gf, bars_mlp]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )

    out_path = RESULTS_DIR / "gf_vs_mlp_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved GF vs MLP comparison plot: {out_path}\n")


# ---------------------------------------------------------------------------
# 3) Per-feature R2 bar chart (GABA R1, GF)
# ---------------------------------------------------------------------------
def plot_per_feature_r2() -> None:
    """Per-feature R2 bar chart for GABA R1 GF results (cv10)."""
    result = load_result("gaba", 1, "cv10", "gf")
    if result is None:
        print("Skipping per-feature R2 plot -- gaba/round1/cv10_gf_results.json not found.\n")
        return

    pf_r2 = extract_per_feature_r2(result)
    if not pf_r2:
        print("Skipping per-feature R2 plot -- no per_feature_r2 data in results.\n")
        return

    # Sort features by R2 descending
    sorted_features = sorted(pf_r2.items(), key=lambda kv: kv[1], reverse=True)
    names = [kv[0] for kv in sorted_features]
    values = [kv[1] for kv in sorted_features]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in values]
    ax.barh(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean R$^2$ across folds", fontsize=12)
    ax.set_title("GABA Round 1 -- Per-Feature R$^2$ (Geneformer, 10-fold CV)", fontsize=13)
    ax.invert_yaxis()  # highest R2 on top
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

    out_path = RESULTS_DIR / "gaba_per_feature_r2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved per-feature R2 plot: {out_path}\n")


# ---------------------------------------------------------------------------
# 4) Cross-species comparison table
# ---------------------------------------------------------------------------
def print_cross_species_table(rows: list[dict]) -> None:
    """
    Compare human Patch-seq results against hardcoded mouse baselines.
    Uses GABA R1 cv10 as the primary human comparison point.
    """
    print("=" * 80)
    print("CROSS-SPECIES COMPARISON: Human Patch-seq vs Mouse (literature)")
    print("=" * 80)

    # Gather human results for GABA R1, cv10
    human_results = {}
    for r in rows:
        if r["dataset"] == "gaba" and r["round"] == 1 and r["cv_type"] == "cv10":
            human_results[r["model"]] = r["mean_r2"]

    print(f"\n{'Region/Species':25s} {'GF R2':>10s} {'MLP R2':>10s} {'Delta':>10s}")
    print("-" * 60)

    # Mouse baselines
    for region, vals in sorted(MOUSE_BASELINES.items()):
        gf = vals["GF"]
        mlp = vals["MLP"]
        delta = gf - mlp
        print(f"Mouse {region:20s} {gf:10.3f} {mlp:10.3f} {delta:+10.3f}")

    # Human
    h_gf = human_results.get("GF")
    h_mlp = human_results.get("MLP")
    gf_str = f"{h_gf:.3f}" if h_gf is not None else "N/A"
    mlp_str = f"{h_mlp:.3f}" if h_mlp is not None else "N/A"
    if h_gf is not None and h_mlp is not None:
        delta_str = f"{h_gf - h_mlp:+.3f}"
    else:
        delta_str = "N/A"
    print(f"Human GABA R1 (cv10)      {gf_str:>10s} {mlp_str:>10s} {delta_str:>10s}")

    print("-" * 60)
    print(
        "Note: Mouse values are hardcoded from published results "
        "(M1 GF=0.435, MLP=0.365; V1 GF=0.345, MLP=0.295)."
    )
    if h_gf is None and h_mlp is None:
        print("      Human results not yet available -- run training scripts first.")
    print()


# ---------------------------------------------------------------------------
# 5) LODO donor-level boxplot
# ---------------------------------------------------------------------------
def plot_lodo_comparison() -> None:
    """
    Boxplot of per-donor / per-fold R2 for LODO experiments, comparing
    GF and MLP side-by-side for each dataset.
    """
    # Collect all LODO results
    plot_data = []  # list of (label, model, fold_r2s)
    for dataset in DATASETS:
        for round_num in ROUNDS[dataset]:
            for model in MODEL_TYPES:
                result = load_result(dataset, round_num, "lodo", model)
                if result is None:
                    continue
                fold_r2s = extract_fold_r2s(result)
                if fold_r2s is None or len(fold_r2s) == 0:
                    continue
                label = f"{dataset} R{round_num}"
                plot_data.append(
                    {
                        "label": label,
                        "model": model.upper(),
                        "fold_r2s": fold_r2s,
                    }
                )

    if not plot_data:
        print("Skipping LODO boxplot -- no LODO results found.\n")
        return

    # Organise data for grouped boxplot
    labels = sorted(set(d["label"] for d in plot_data))
    models = ["GF", "MLP"]
    colors = {"GF": "#4C72B0", "MLP": "#DD8452"}

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.5), 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    positions = []
    tick_positions = []
    tick_labels = []

    pos = 0
    for lab in labels:
        group_start = pos
        for model in models:
            match = [d for d in plot_data if d["label"] == lab and d["model"] == model]
            if match:
                bp = ax.boxplot(
                    match[0]["fold_r2s"],
                    positions=[pos],
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[model], alpha=0.7),
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3),
                )
            pos += 1
        tick_positions.append((group_start + pos - 1) / 2)
        tick_labels.append(lab)
        pos += 0.5  # gap between groups

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("R$^2$ per donor (LODO fold)", fontsize=12)
    ax.set_title("Leave-One-Donor-Out CV: Per-Donor R$^2$ Distribution", fontsize=13)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors["GF"], alpha=0.7, edgecolor="black", label="Geneformer"),
        Patch(facecolor=colors["MLP"], alpha=0.7, edgecolor="black", label="MLP (IC genes)"),
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    out_path = RESULTS_DIR / "lodo_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved LODO comparison plot: {out_path}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Scanning for result files...\n")

    rows, missing = scan_all_results()

    if missing:
        print(f"Missing result files ({len(missing)}):")
        for m in missing:
            print(m)
        print()

    if rows:
        print(f"Found {len(rows)} result file(s).\n")
    else:
        print("No result files found yet. Run training scripts (04, 05) first.\n")

    # 1) Summary table
    print_summary_table(rows)

    # 2) GF vs MLP bar chart
    plot_gf_vs_mlp(rows)

    # 3) Per-feature R2 (GABA R1 GF)
    plot_per_feature_r2()

    # 4) Cross-species comparison
    print_cross_species_table(rows)

    # 5) LODO boxplot
    plot_lodo_comparison()

    print("Analysis complete.")


if __name__ == "__main__":
    main()
