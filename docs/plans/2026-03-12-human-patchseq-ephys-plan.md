# Human Patch-seq Ephys Prediction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fine-tune Geneformer V1-10M to predict electrophysiology features from human Patch-seq transcriptomes, comparing GABAergic (704 cells) and excitatory (304 cells) neurons independently and pooled.

**Architecture:** GF-V1 (6L, 256d BERT) with gene-level mean pool → regression head (Linear 256→N). Top2 freeze (layers 4-5 trainable). MLP baseline on ~140 ion channel genes for comparison.

**Tech Stack:** Python 3.10+, R 4.x, conda env `geneformer`, PyTorch, HuggingFace Transformers, anndata, scanpy, geneformer package.

**Design doc:** `docs/plans/2026-03-12-human-patchseq-ephys-design.md`

---

## Task 1: Extract Gene Expression from RData to h5ad

**Files:**
- Create: `PatchSeq/scripts/01_extract_rdata.R`
- Output: `/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/gaba_expression.h5ad`
- Output: `/home/jw3514/Work/NeurSim/patchseq_human_L23/data/excitatory_expression.h5ad`

**Context:**
- GABA data: Two RData files at `/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/complete_patchseq_data_sets{1,2}.RData`. Must merge with `cbind(datPatch1, datPatch2)`. Contains `datPatch` (genes × cells expression matrix), `annoPatch` (gene annotations), `metaPatch` (cell metadata).
- Excitatory data: Single RData at `/home/jw3514/Work/NeurSim/patchseq_human_L23/data/input_patchseq_data_sets.RData`. Format TBD — inspect and extract similarly.
- Both need: expression matrix with Ensembl gene IDs as var index, cell IDs matching ephys specimen_id, `n_counts` in obs.

**Step 1: Write the R extraction script**

```r
#!/usr/bin/env Rscript
# 01_extract_rdata.R — Extract gene expression from RData to h5ad
# Requires: anndata (R package), or exports to CSV for Python conversion

library(Matrix)

# ── GABA dataset (Lee & Dalley 2023) ───────────────────────────────
cat("=== Extracting GABA dataset ===\n")
gaba_dir <- "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data"

load(file.path(gaba_dir, "complete_patchseq_data_sets1.RData"))
load(file.path(gaba_dir, "complete_patchseq_data_sets2.RData"))
datPatch <- cbind(datPatch1, datPatch2)

cat(sprintf("  datPatch: %d genes x %d cells\n", nrow(datPatch), ncol(datPatch)))
cat(sprintf("  annoPatch: %d x %d\n", nrow(annoPatch), ncol(annoPatch)))
cat(sprintf("  metaPatch: %d x %d\n", nrow(metaPatch), ncol(metaPatch)))

# Inspect row/column names and annotation structure
cat("  Gene ID sample:", head(rownames(datPatch), 3), "\n")
cat("  Cell ID sample:", head(colnames(datPatch), 3), "\n")
cat("  annoPatch columns:", paste(colnames(annoPatch), collapse=", "), "\n")
cat("  metaPatch columns:", paste(colnames(metaPatch), collapse=", "), "\n")

# Save as CSV for Python conversion (safer than R anndata package)
# Expression matrix: genes x cells (transpose later in Python)
write.csv(as.matrix(datPatch), file.path(gaba_dir, "gaba_expression_matrix.csv"))
write.csv(annoPatch, file.path(gaba_dir, "gaba_gene_annotations.csv"))
write.csv(metaPatch, file.path(gaba_dir, "gaba_cell_metadata.csv"))
cat("  Saved GABA CSVs\n")

# ── Excitatory dataset (Berg et al. 2021) ──────────────────────────
cat("\n=== Extracting Excitatory dataset ===\n")
exc_dir <- "/home/jw3514/Work/NeurSim/patchseq_human_L23/data"
load(file.path(exc_dir, "input_patchseq_data_sets.RData"))

# Inspect what objects were loaded
cat("  Objects loaded:", paste(ls(), collapse=", "), "\n")
# Expected: some combination of expression matrix + metadata
# Print structure of each loaded object to determine format
for (obj_name in ls()) {
    obj <- get(obj_name)
    if (is.matrix(obj) || is.data.frame(obj) || inherits(obj, "dgCMatrix")) {
        cat(sprintf("  %s: %d x %d (%s)\n", obj_name, nrow(obj), ncol(obj), class(obj)[1]))
    }
}

# NOTE: The exact variable names inside the RData are unknown.
# Run this script first in interactive R to inspect, then adjust the export code.
# Placeholder: save whatever expression matrix is found
cat("\n  >>> INSPECT OUTPUT ABOVE to determine variable names <<<\n")
cat("  >>> Then update this script to export the correct objects <<<\n")
```

**Step 2: Run R extraction (inspect first)**

```bash
conda run -n geneformer Rscript PatchSeq/scripts/01_extract_rdata.R
```

Expected: Script prints dimensions and column names. May need adjustment based on actual RData contents, especially for the excitatory dataset.

**Step 3: Write Python h5ad conversion script**

```python
# Add to 01_extract_rdata.R or create 01b_csv_to_h5ad.py
# After CSVs are exported from R, convert to proper h5ad

import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp

# ── GABA ──
gaba_dir = "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data"
expr = pd.read_csv(f"{gaba_dir}/gaba_expression_matrix.csv", index_col=0)
# expr is genes x cells → transpose to cells x genes for AnnData
expr_t = expr.T

# Gene annotations
gene_anno = pd.read_csv(f"{gaba_dir}/gaba_gene_annotations.csv", index_col=0)
# Cell metadata
cell_meta = pd.read_csv(f"{gaba_dir}/gaba_cell_metadata.csv", index_col=0)

# Need ensembl_id in var and n_counts in obs
# Determine how Ensembl IDs are stored (rownames of datPatch or in annoPatch)
# Add n_counts = sum of counts per cell
obs = pd.DataFrame(index=expr_t.index)
obs["n_counts"] = expr_t.sum(axis=1)
# Merge cell metadata by matching index (cell IDs)

var = pd.DataFrame(index=expr_t.columns)
var["ensembl_id"] = var.index  # If rownames are already Ensembl IDs
# If rownames are gene symbols, map via annoPatch

adata = ad.AnnData(
    X=sp.csr_matrix(expr_t.values.astype(np.float32)),
    obs=obs,
    var=var,
)
adata.write_h5ad(f"{gaba_dir}/gaba_expression.h5ad")
```

**Step 4: Verify h5ad files**

```bash
conda run -n geneformer python3 -c "
import anndata as ad
for name, path in [('GABA', '/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/gaba_expression.h5ad'),
                   ('Exc', '/home/jw3514/Work/NeurSim/patchseq_human_L23/data/excitatory_expression.h5ad')]:
    a = ad.read_h5ad(path)
    print(f'{name}: {a.shape[0]} cells x {a.shape[1]} genes')
    print(f'  obs columns: {list(a.obs.columns)}')
    print(f'  var columns: {list(a.var.columns)}')
    print(f'  ensembl_id sample: {a.var[\"ensembl_id\"].head(3).tolist()}')
    print(f'  n_counts range: {a.obs[\"n_counts\"].min():.0f} - {a.obs[\"n_counts\"].max():.0f}')
    print()
"
```

Expected: GABA ~778 cells, Exc ~385 cells. Both have `ensembl_id` in var and `n_counts` in obs.

**Step 5: Commit**

```bash
git add PatchSeq/scripts/01_extract_rdata.R PatchSeq/scripts/01b_csv_to_h5ad.py
git commit -m "Add RData to h5ad extraction for human Patch-seq datasets"
```

**Notes:**
- The RData format is unknown until inspected. This task requires interactive iteration.
- The GABA dataset gene IDs may be gene symbols (not Ensembl). Check `annoPatch` for an Ensembl mapping column.
- The excitatory RData (`input_patchseq_data_sets.RData`) structure is completely unknown — inspect first.
- Large CSV export (~778 cells × ~50K genes) may take a few GB. Consider sparse matrix export if needed.

---

## Task 2: Tokenize with Geneformer V1

**Files:**
- Create: `PatchSeq/scripts/02_tokenize.py`
- Output: `PatchSeq/data/gaba_tokenized.dataset/`
- Output: `PatchSeq/data/excitatory_tokenized.dataset/`

**Context:**
- Geneformer V1 tokenizer: `TranscriptomeTokenizer(model_version="V1")` automatically uses 30M dictionaries, max 2048 tokens, no special tokens (no CLS/EOS).
- V1 token dictionary: 25,426 human Ensembl IDs at `geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl`
- Input h5ad needs: `ensembl_id` in `.var`, `n_counts` in `.obs`
- Custom metadata to preserve: `specimen_id` (for ephys alignment), `donor` (for LODO CV)

**Step 1: Write tokenization script**

```python
"""02_tokenize.py — Tokenize human Patch-seq h5ad files with Geneformer V1."""
import os
import sys

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from geneformer import TranscriptomeTokenizer

DATASETS = {
    "gaba": {
        "h5ad_dir": "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data",
        "h5ad_name": "gaba_expression.h5ad",
    },
    "excitatory": {
        "h5ad_dir": "/home/jw3514/Work/NeurSim/patchseq_human_L23/data",
        "h5ad_name": "excitatory_expression.h5ad",
    },
}

OUTPUT_DIR = "/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, info in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Tokenizing {name} dataset")
    print(f"{'='*60}")

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"specimen_id": "specimen_id"},  # preserve cell ID
        nproc=10,
        model_input_size=2048,
        model_version="V1",
    )

    tk.tokenize_data(
        data_directory=info["h5ad_dir"],
        output_directory=OUTPUT_DIR,
        output_prefix=f"{name}_tokenized",
        file_format="h5ad",
    )

    print(f"  Output: {OUTPUT_DIR}/{name}_tokenized.dataset")
```

**Step 2: Run tokenization**

```bash
conda run -n geneformer python PatchSeq/scripts/02_tokenize.py
```

Expected: Two HuggingFace datasets with fields `input_ids`, `length`, and `specimen_id`.

**Step 3: Verify tokenized data**

```bash
conda run -n geneformer python3 -c "
from datasets import load_from_disk
for name in ['gaba', 'excitatory']:
    ds = load_from_disk(f'/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data/{name}_tokenized.dataset')
    print(f'{name}: {len(ds)} cells, features: {ds.column_names}')
    print(f'  input_ids length range: {min(ds[\"length\"])}-{max(ds[\"length\"])}')
    print(f'  specimen_id sample: {ds[\"specimen_id\"][:3]}')
"
```

Expected: GABA ~778 cells, Exc ~385 cells. Each with `input_ids`, `length`, `specimen_id`.

**Step 4: Commit**

```bash
git add PatchSeq/scripts/02_tokenize.py
git commit -m "Add V1 tokenization script for human Patch-seq data"
```

**Notes:**
- The tokenizer processes ALL h5ad files in the directory. If there are multiple h5ad files, use `input_identifier` to filter, or ensure only the target h5ad is in the directory.
- V1 has no CLS/EOS tokens — all positions are gene tokens. This matters for mean pooling later.
- Some cells may be filtered out if they have too few genes matching the V1 token dictionary. Check output count vs input count.

---

## Task 3: Prepare Ephys Features

**Files:**
- Create: `PatchSeq/scripts/03_prepare_ephys.py`
- Output: `PatchSeq/data/gaba_ephys_r1.pkl` (Round 1: 18 matched features)
- Output: `PatchSeq/data/gaba_ephys_r2.pkl` (Round 2: ~80 extended features)
- Output: `PatchSeq/data/excitatory_ephys.pkl` (18 features)

**Context:**
- GABA ephys: `/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/LeeDalley_ephys_fx.csv` — 704 cells × 93 features
- Excitatory ephys: `/home/jw3514/Work/NeurSim/patchseq_human_L23/data/human_mouse_ephys_all_0127.csv` — 447 cells × 18 features (mixed human+mouse, filter to human)
- GABA metadata: `/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/LeeDalley_manuscript_metadata.csv` — has `specimen_id`, `Donor`, `Has_ephys`, `condition`, `Revised_subclass_label`
- Excitatory metadata: `/home/jw3514/Work/NeurSim/patchseq_human_L23/data/human_IVSCC_excitatory_L23_consolidated_0131.csv` — has `SpecimenID`, `donor`, `rna_amplification_call`, `SeuratMapping`

**Step 1: Write ephys preparation script**

```python
"""03_prepare_ephys.py — Load, align, and save ephys features for both datasets."""
import pickle
import numpy as np
import pandas as pd

# ── Feature mapping: Berg (long_square) → GABA (hero) equivalents ──
FEATURE_MAP_BERG_TO_GABA = {
    "adapt_mean": "adapt_mean",
    "avg_rate_hero": "avg_rate_hero",
    "downstroke_long_square": "downstroke_hero",
    "fast_trough_v_long_square_rel": "fast_trough_deltav_hero",
    "fi_fit_slope": "fi_fit_slope",
    "first_isi_hero_inv": "first_isi_inv_hero",
    "input_resistance": "input_resistance",
    "latency_rheo": "latency_rheo",
    "peak_v_long_square_rel": "peak_deltav_hero",
    "rheobase_i": "rheobase_i",
    "sag": "sag",
    "tau": "tau",
    "threshold_v_long_square": "threshold_v_hero",
    "trough_v_long_square_rel": "trough_deltav_hero",
    "upstroke_downstroke_ratio_long_square": "upstroke_downstroke_ratio_hero",
    "upstroke_long_square": "upstroke_hero",
    "v_baseline": "v_baseline",
    "width_long_square": "width_hero",
}
SHARED_FEATURE_NAMES = list(FEATURE_MAP_BERG_TO_GABA.keys())  # Use Berg names as canonical

OUT_DIR = "/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data"

# ── GABA dataset ───────────────────────────────────────────────────
print("=== GABA ephys ===")
gaba_ephys = pd.read_csv(
    "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/LeeDalley_ephys_fx.csv"
)
gaba_meta = pd.read_csv(
    "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/LeeDalley_manuscript_metadata.csv"
)
gaba_ephys = gaba_ephys.set_index("specimen_id")

# Round 1: 18 shared features (renamed to canonical Berg names)
gaba_r1_cols = list(FEATURE_MAP_BERG_TO_GABA.values())
gaba_r1 = gaba_ephys[gaba_r1_cols].copy()
gaba_r1.columns = SHARED_FEATURE_NAMES  # Rename to canonical names

# Drop rows with any missing in the 18 features
n_before = len(gaba_r1)
gaba_r1 = gaba_r1.dropna()
print(f"  Round 1: {n_before} → {len(gaba_r1)} cells (dropped {n_before - len(gaba_r1)} with NaN)")

# Round 2: All features with >80% coverage, drop cells with >20% missing
all_cols = [c for c in gaba_ephys.columns if c != "specimen_id"]
coverage = gaba_ephys[all_cols].notna().mean()
r2_cols = coverage[coverage > 0.8].index.tolist()
gaba_r2 = gaba_ephys[r2_cols].copy()
missing_frac = gaba_r2.isna().mean(axis=1)
gaba_r2 = gaba_r2[missing_frac < 0.2].dropna()
print(f"  Round 2: {len(gaba_ephys)} → {len(gaba_r2)} cells, {len(r2_cols)} features")

# Add donor info for LODO CV
gaba_donor_map = dict(zip(gaba_meta["specimen_id"], gaba_meta["Donor"]))

# ── Excitatory dataset ─────────────────────────────────────────────
print("\n=== Excitatory ephys ===")
exc_ephys = pd.read_csv(
    "/home/jw3514/Work/NeurSim/patchseq_human_L23/data/human_mouse_ephys_all_0127.csv"
)
exc_meta = pd.read_csv(
    "/home/jw3514/Work/NeurSim/patchseq_human_L23/data/human_IVSCC_excitatory_L23_consolidated_0131.csv"
)
exc_ephys = exc_ephys.set_index("specimen_id")

# Filter to human cells with RNA Pass quality
human_pass_ids = set(exc_meta[exc_meta["rna_amplification_call"] == "Pass"]["SpecimenID"])
exc_ephys = exc_ephys[exc_ephys.index.isin(human_pass_ids)]

# Use Berg's 18 features directly (already in canonical names)
exc_r1 = exc_ephys[SHARED_FEATURE_NAMES].copy()
n_before = len(exc_r1)
exc_r1 = exc_r1.dropna()
print(f"  Round 1: {n_before} → {len(exc_r1)} cells (dropped {n_before - len(exc_r1)} with NaN)")

# Donor map
exc_donor_map = dict(zip(exc_meta["SpecimenID"], exc_meta["donor"]))

# ── Save ───────────────────────────────────────────────────────────
import os
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{OUT_DIR}/gaba_ephys_r1.pkl", "wb") as f:
    pickle.dump({"ephys": gaba_r1, "donor_map": gaba_donor_map, "feature_names": SHARED_FEATURE_NAMES}, f)

with open(f"{OUT_DIR}/gaba_ephys_r2.pkl", "wb") as f:
    pickle.dump({"ephys": gaba_r2, "donor_map": gaba_donor_map, "feature_names": r2_cols}, f)

with open(f"{OUT_DIR}/excitatory_ephys.pkl", "wb") as f:
    pickle.dump({"ephys": exc_r1, "donor_map": exc_donor_map, "feature_names": SHARED_FEATURE_NAMES}, f)

print(f"\nSaved to {OUT_DIR}/")
print(f"  gaba_ephys_r1.pkl: {gaba_r1.shape}")
print(f"  gaba_ephys_r2.pkl: {gaba_r2.shape}")
print(f"  excitatory_ephys.pkl: {exc_r1.shape}")
```

**Step 2: Run ephys preparation**

```bash
conda run -n geneformer python PatchSeq/scripts/03_prepare_ephys.py
```

Expected: Three pickle files with aligned ephys DataFrames (index=specimen_id) and donor maps.

**Step 3: Commit**

```bash
git add PatchSeq/scripts/03_prepare_ephys.py
git commit -m "Add ephys feature preparation with cross-dataset alignment"
```

---

## Task 4: Fine-tune Geneformer V1 with Ephys Regression Head

**Files:**
- Create: `PatchSeq/scripts/04_finetune_gf_ephys.py`
- Output: `PatchSeq/results/{gaba,excitatory,pooled}/` — JSON results, saved models

**Context:**
- Model: `models/Geneformer/Geneformer-V1-10M/` — BertForMaskedLM, 6L, 256d, vocab 25426
- Load as `BertForSequenceClassification` with `num_labels=N`, `problem_type="regression"`
- Top2 freeze: freeze embeddings + layers 0-3, fine-tune layers 4-5 + classifier head
- V1 has no CLS token. `BertForSequenceClassification` uses `BertPooler` which takes the first token's hidden state. For V1 (no CLS), the first token is just the highest-ranked gene. This matches the mouse approach where BertPooler is used directly.
- Custom collator needed for multi-target float regression labels
- Two CV modes: 10-fold and leave-one-donor-out

**Step 1: Write the fine-tuning script**

This is the main script (~350 lines). Key components:

```python
"""04_finetune_gf_ephys.py — Fine-tune GF-V1 for ephys regression."""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from transformers import (
    BertForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")
from geneformer.collator_for_classification import DataCollatorForCellClassification

# ── Paths ──────────────────────────────────────────────────────────
MODEL_DIR = "/home/jw3514/Work/Geneformer/Geneformer/models/Geneformer/Geneformer-V1-10M"
DATA_DIR = "/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data"
RESULTS_DIR = "/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/results"

# ── Custom collator for regression ─────────────────────────────────
class DataCollatorForEphysRegression(DataCollatorForCellClassification):
    def __call__(self, features):
        batch = self._prepare_batch(features)
        result = {}
        for k, v in batch.items():
            if k == "labels":
                result[k] = (
                    v if isinstance(v, torch.Tensor)
                    else torch.tensor(v, dtype=torch.float32)
                )
            else:
                result[k] = (
                    v.to(torch.int64) if isinstance(v, torch.Tensor)
                    else torch.tensor(v, dtype=torch.int64)
                )
        return result

# ── Model loading with freeze ──────────────────────────────────────
def load_model(n_targets, freeze_mode="top2"):
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=n_targets,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )
    n_layers = model.config.num_hidden_layers  # 6

    if freeze_mode == "top2":
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(n_layers - 2):  # freeze layers 0-3
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
    return model

# ── Metrics ────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    r2 = r2_score(labels, preds, multioutput="uniform_average")
    n_feat = labels.shape[1]
    pearson_rs = []
    for j in range(n_feat):
        mask = np.isfinite(labels[:, j]) & np.isfinite(preds[:, j])
        if mask.sum() > 2:
            r, _ = pearsonr(labels[mask, j], preds[mask, j])
            pearson_rs.append(r if np.isfinite(r) else 0.0)
        else:
            pearson_rs.append(0.0)
    return {"r2": r2, "mean_pearson_r": np.mean(pearson_rs)}

# ── Data alignment ─────────────────────────────────────────────────
def align_data(tokenized_ds, ephys_df):
    """Align tokenized cells with ephys by specimen_id. Returns matched lists."""
    tok_ids = tokenized_ds["specimen_id"]
    # Handle specimen_id type mismatch (str vs int)
    tok_id_map = {}
    for i, cid in enumerate(tok_ids):
        tok_id_map[str(cid)] = i
        tok_id_map[int(cid) if str(cid).isdigit() else cid] = i

    common_ids = [sid for sid in ephys_df.index if sid in tok_id_map or str(sid) in tok_id_map]
    print(f"  Aligned: {len(common_ids)} cells (tok={len(tok_ids)}, ephys={len(ephys_df)})")

    tok_indices = [tok_id_map.get(sid, tok_id_map.get(str(sid))) for sid in common_ids]
    input_ids_all = [tokenized_ds[i]["input_ids"] for i in tok_indices]
    lengths_all = [tokenized_ds[i]["length"] for i in tok_indices]
    ephys_values = ephys_df.loc[common_ids].values.astype(np.float32)
    donor_ids = common_ids  # For LODO, map these through donor_map

    return input_ids_all, lengths_all, ephys_values, common_ids

# ── Build train/test data ──────────────────────────────────────────
def make_split(input_ids_all, lengths_all, ephys_values, train_idx, test_idx):
    """Z-score ephys on train, build data dicts."""
    train_ephys = ephys_values[train_idx]
    mean = train_ephys.mean(axis=0)
    std = train_ephys.std(axis=0)
    std[std == 0] = 1.0
    normed = (ephys_values - mean) / std

    def make_data(idx_arr):
        return [
            {
                "input_ids": input_ids_all[i],
                "label": normed[i].tolist(),
                "length": lengths_all[i],
            }
            for i in idx_arr
        ]

    return make_data(train_idx), make_data(test_idx), {"mean": mean, "std": std}

# ── Training function ──────────────────────────────────────────────
def train_fold(train_data, test_data, n_targets, run_dir, fold_name="fold"):
    model = load_model(n_targets, freeze_mode="top2")

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorForEphysRegression(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    preds = trainer.predict(test_data)
    return metrics, preds.predictions, preds.label_ids

# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gaba", "excitatory", "pooled"], required=True)
    parser.add_argument("--round", type=int, default=1, choices=[1, 2])
    parser.add_argument("--cv", type=int, default=10, help="Number of CV folds (0=LODO)")
    args = parser.parse_args()

    # Load tokenized data
    if args.dataset == "pooled":
        ds_gaba = load_from_disk(f"{DATA_DIR}/gaba_tokenized.dataset")
        ds_exc = load_from_disk(f"{DATA_DIR}/excitatory_tokenized.dataset")
        # Concatenate datasets (both must have same columns)
        from datasets import concatenate_datasets
        ds = concatenate_datasets([ds_gaba, ds_exc])
    else:
        ds = load_from_disk(f"{DATA_DIR}/{args.dataset}_tokenized.dataset")

    # Load ephys
    if args.dataset == "gaba":
        ephys_file = f"gaba_ephys_r{args.round}.pkl"
    elif args.dataset == "excitatory":
        ephys_file = "excitatory_ephys.pkl"
    else:  # pooled
        # Merge GABA r1 and excitatory (both have 18 shared features)
        with open(f"{DATA_DIR}/gaba_ephys_r1.pkl", "rb") as f:
            gaba_data = pickle.load(f)
        with open(f"{DATA_DIR}/excitatory_ephys.pkl", "rb") as f:
            exc_data = pickle.load(f)
        pooled_ephys = pd.concat([gaba_data["ephys"], exc_data["ephys"]])
        pooled_donors = {**gaba_data["donor_map"], **exc_data["donor_map"]}
        ephys_file = None  # handled inline

    if ephys_file:
        with open(f"{DATA_DIR}/{ephys_file}", "rb") as f:
            data = pickle.load(f)
        ephys_df = data["ephys"]
        donor_map = data["donor_map"]
        feature_names = data["feature_names"]
    else:
        ephys_df = pooled_ephys
        donor_map = pooled_donors
        feature_names = gaba_data["feature_names"]

    import pandas as pd
    n_targets = len(feature_names)
    print(f"Dataset: {args.dataset}, Round: {args.round}, Features: {n_targets}")

    # Align
    input_ids_all, lengths_all, ephys_values, cell_ids = align_data(ds, ephys_df)
    n_cells = len(cell_ids)

    # Output dir
    out_dir = Path(RESULTS_DIR) / args.dataset / f"round{args.round}"
    os.makedirs(out_dir, exist_ok=True)

    # ── CV ──────────────────────────────────────────────────────────
    if args.cv > 0:
        # K-fold CV
        kf = KFold(n_splits=args.cv, shuffle=True, random_state=42)
        all_preds = np.full((n_cells, n_targets), np.nan)
        all_labels = np.full((n_cells, n_targets), np.nan)
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(range(n_cells))):
            print(f"\n--- Fold {fold_i+1}/{args.cv} (train={len(train_idx)}, test={len(test_idx)}) ---")
            train_data, test_data, scaler = make_split(
                input_ids_all, lengths_all, ephys_values, train_idx, test_idx
            )
            fold_dir = out_dir / f"cv{args.cv}" / f"fold{fold_i}"
            os.makedirs(fold_dir, exist_ok=True)

            metrics, preds, labels = train_fold(train_data, test_data, n_targets, fold_dir)
            fold_metrics.append(metrics)
            all_preds[test_idx] = preds
            all_labels[test_idx] = labels
            print(f"  Fold {fold_i+1} R²={metrics['eval_r2']:.4f}, Pearson r={metrics['eval_mean_pearson_r']:.4f}")

        # Global metrics from concatenated out-of-fold predictions
        valid = ~np.isnan(all_preds[:, 0])
        global_r2 = r2_score(all_labels[valid], all_preds[valid], multioutput="uniform_average")
        per_feat_r2 = [r2_score(all_labels[valid, j], all_preds[valid, j]) for j in range(n_targets)]

        results = {
            "dataset": args.dataset,
            "round": args.round,
            "cv": args.cv,
            "n_cells": n_cells,
            "n_features": n_targets,
            "feature_names": feature_names,
            "global_r2": float(global_r2),
            "mean_fold_r2": float(np.mean([m["eval_r2"] for m in fold_metrics])),
            "std_fold_r2": float(np.std([m["eval_r2"] for m in fold_metrics])),
            "per_feature_r2": {feature_names[j]: float(per_feat_r2[j]) for j in range(n_targets)},
            "fold_metrics": fold_metrics,
        }

        with open(out_dir / f"cv{args.cv}_gf_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nGlobal R²={global_r2:.4f}, Mean fold R²={results['mean_fold_r2']:.4f} ± {results['std_fold_r2']:.4f}")

    else:
        # Leave-one-donor-out CV
        donors = [donor_map.get(cid, donor_map.get(str(cid), "unknown")) for cid in cell_ids]
        unique_donors = sorted(set(donors))
        print(f"LODO CV: {len(unique_donors)} donors")

        all_preds = np.full((n_cells, n_targets), np.nan)
        all_labels = np.full((n_cells, n_targets), np.nan)
        fold_metrics = []

        for donor in unique_donors:
            test_idx = np.array([i for i, d in enumerate(donors) if d == donor])
            train_idx = np.array([i for i, d in enumerate(donors) if d != donor])
            if len(test_idx) < 2:
                continue
            print(f"\n--- LODO: held-out donor {donor} ({len(test_idx)} cells) ---")
            train_data, test_data, scaler = make_split(
                input_ids_all, lengths_all, ephys_values, train_idx, test_idx
            )
            fold_dir = out_dir / "lodo" / f"donor_{donor}"
            os.makedirs(fold_dir, exist_ok=True)

            metrics, preds, labels = train_fold(train_data, test_data, n_targets, fold_dir)
            fold_metrics.append({"donor": donor, "n_cells": len(test_idx), **metrics})
            all_preds[test_idx] = preds
            all_labels[test_idx] = labels
            print(f"  Donor {donor} R²={metrics['eval_r2']:.4f}")

        valid = ~np.isnan(all_preds[:, 0])
        global_r2 = r2_score(all_labels[valid], all_preds[valid], multioutput="uniform_average")

        results = {
            "dataset": args.dataset,
            "round": args.round,
            "cv": "lodo",
            "n_cells": int(valid.sum()),
            "n_features": n_targets,
            "feature_names": feature_names,
            "global_r2": float(global_r2),
            "donor_metrics": fold_metrics,
        }
        with open(out_dir / "lodo_gf_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nLODO Global R²={global_r2:.4f}")


if __name__ == "__main__":
    main()
```

**Step 2: Run experiments**

```bash
# Phase 1A: GABA, Round 1, 10-fold
conda run -n geneformer python PatchSeq/scripts/04_finetune_gf_ephys.py --dataset gaba --round 1 --cv 10

# Phase 1A: GABA, Round 1, LODO
conda run -n geneformer python PatchSeq/scripts/04_finetune_gf_ephys.py --dataset gaba --round 1 --cv 0

# Phase 1B: GABA, Round 2 (extended features)
conda run -n geneformer python PatchSeq/scripts/04_finetune_gf_ephys.py --dataset gaba --round 2 --cv 10

# Phase 1C: Excitatory
conda run -n geneformer python PatchSeq/scripts/04_finetune_gf_ephys.py --dataset excitatory --round 1 --cv 10
conda run -n geneformer python PatchSeq/scripts/04_finetune_gf_ephys.py --dataset excitatory --round 1 --cv 0

# Phase 2: Pooled
conda run -n geneformer python PatchSeq/scripts/04_finetune_gf_ephys.py --dataset pooled --round 1 --cv 10
```

**Step 3: Commit**

```bash
git add PatchSeq/scripts/04_finetune_gf_ephys.py
git commit -m "Add GF-V1 ephys fine-tuning with 10-fold and LODO CV"
```

**Notes:**
- `BertForSequenceClassification` uses BertPooler (first token → Linear(256,256) + Tanh → classifier). In V1 (no CLS), the first token is the highest-ranked gene. This is equivalent to the mouse approach.
- The mouse script also had pseudogene filtering (Gm#### genes). Human data shouldn't have mouse pseudogenes, but verify no unexpected tokens.
- `ignore_mismatched_sizes=True` is needed because we're loading a MaskedLM model as SequenceClassification.
- Early stopping patience=5 on eval R². Each fold runs up to 20 epochs.
- LODO CV will have highly variable test set sizes. Skip donors with <2 cells.

---

## Task 5: MLP Baseline on Ion Channel Genes

**Files:**
- Create: `PatchSeq/scripts/05_mlp_baseline.py`
- Output: `PatchSeq/results/{gaba,excitatory,pooled}/` — JSON results alongside GF results

**Context:**
- Architecture: `Linear(N_ic, 256) → Tanh → Dropout(0.1) → Linear(256, N_targets)`
- Input: log2(CPM+1) expression of ~140 human ion channel genes
- Human IC genes: SCN1A-SCN11A, KCNA1-KCNV2, CACNA1A-CACNA1S, HCN1-4, CLCN1-7 + markers PVALB, SST, VIP, LAMP5
- Same CV protocol: 10-fold and LODO
- Same z-scoring: ephys targets z-scored on train, IC expression z-scored on train

**Step 1: Write MLP baseline script**

```python
"""05_mlp_baseline.py — MLP baseline on ion channel genes for ephys prediction."""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# ── Ion channel gene prefixes (human) ──────────────────────────────
IC_PREFIXES = [
    "SCN1", "SCN2", "SCN3", "SCN4", "SCN5", "SCN7", "SCN8", "SCN9", "SCN10", "SCN11",
    "KCNA", "KCNB", "KCNC", "KCND", "KCNE", "KCNF", "KCNG", "KCNH", "KCNJ",
    "KCNK", "KCNMA", "KCNMB", "KCNN", "KCNQ", "KCNS", "KCNT", "KCNU", "KCNV",
    "CACNA", "CACNB", "CACNG",
    "HCN",
    "CLCN",
]
MARKER_GENES = ["PVALB", "SST", "VIP", "LAMP5"]

DATA_DIR = "/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data"
RESULTS_DIR = "/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/results"

# ── MLP model ──────────────────────────────────────────────────────
class EphysMLP(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, n_output),
        )

    def forward(self, x):
        return self.net(x)

# ── Training loop ──────────────────────────────────────────────────
def train_mlp(X_train, y_train, X_test, y_test, n_targets, max_epochs=100, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EphysMLP(X_train.shape[1], n_targets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.float32).to(device)

    best_r2 = -np.inf
    wait = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_te).cpu().numpy()
            test_r2 = r2_score(y_test, test_pred, multioutput="uniform_average")

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = model(X_te).cpu().numpy()
    return final_pred, best_r2

# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gaba", "excitatory", "pooled"], required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--cv", type=int, default=10)
    args = parser.parse_args()

    # Load h5ad for expression
    if args.dataset == "pooled":
        adata_g = ad.read_h5ad("/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/gaba_expression.h5ad")
        adata_e = ad.read_h5ad("/home/jw3514/Work/NeurSim/patchseq_human_L23/data/excitatory_expression.h5ad")
        # Align genes, concatenate
        common_genes = list(set(adata_g.var_names) & set(adata_e.var_names))
        adata = ad.concat([adata_g[:, common_genes], adata_e[:, common_genes]])
    else:
        h5ad_path = {
            "gaba": "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/gaba_expression.h5ad",
            "excitatory": "/home/jw3514/Work/NeurSim/patchseq_human_L23/data/excitatory_expression.h5ad",
        }[args.dataset]
        adata = ad.read_h5ad(h5ad_path)

    # Select IC genes by gene symbol prefix matching
    gene_names = adata.var["gene_symbol"] if "gene_symbol" in adata.var else pd.Series(adata.var_names)
    ic_mask = gene_names.apply(
        lambda g: any(g.upper().startswith(p) for p in IC_PREFIXES) or g.upper() in MARKER_GENES
    )
    ic_genes = gene_names[ic_mask].tolist()
    print(f"Selected {len(ic_genes)} IC + marker genes")

    # Extract expression: log2(CPM + 1)
    X_raw = adata[:, ic_mask].X
    if hasattr(X_raw, "toarray"):
        X_raw = X_raw.toarray()
    lib_size = np.median(adata.obs["n_counts"].values)
    X_cpm = X_raw / adata.obs["n_counts"].values[:, None] * lib_size
    X_expr = np.log2(X_cpm + 1).astype(np.float32)

    # Load ephys (same as Task 4)
    if args.dataset == "gaba":
        ephys_file = f"gaba_ephys_r{args.round}.pkl"
    elif args.dataset == "excitatory":
        ephys_file = "excitatory_ephys.pkl"
    else:
        with open(f"{DATA_DIR}/gaba_ephys_r1.pkl", "rb") as f:
            gaba_data = pickle.load(f)
        with open(f"{DATA_DIR}/excitatory_ephys.pkl", "rb") as f:
            exc_data = pickle.load(f)
        ephys_df = pd.concat([gaba_data["ephys"], exc_data["ephys"]])
        donor_map = {**gaba_data["donor_map"], **exc_data["donor_map"]}
        feature_names = gaba_data["feature_names"]
        ephys_file = None

    if ephys_file:
        with open(f"{DATA_DIR}/{ephys_file}", "rb") as f:
            data = pickle.load(f)
        ephys_df, donor_map, feature_names = data["ephys"], data["donor_map"], data["feature_names"]

    # Align cells by specimen_id
    cell_ids = adata.obs.index if "specimen_id" not in adata.obs else adata.obs["specimen_id"]
    common = [sid for sid in ephys_df.index if sid in set(cell_ids) or str(sid) in set(cell_ids.astype(str))]
    # Map to adata indices
    cell_id_to_idx = {str(cid): i for i, cid in enumerate(cell_ids)}
    adata_idx = [cell_id_to_idx[str(sid)] for sid in common]

    X = X_expr[adata_idx]
    y = ephys_df.loc[common].values.astype(np.float32)
    n_cells, n_targets = len(common), len(feature_names)
    print(f"Aligned: {n_cells} cells, {n_targets} features, {X.shape[1]} IC genes")

    out_dir = Path(RESULTS_DIR) / args.dataset / f"round{args.round}"
    os.makedirs(out_dir, exist_ok=True)

    # ── K-fold CV ──────────────────────────────────────────────────
    if args.cv > 0:
        kf = KFold(n_splits=args.cv, shuffle=True, random_state=42)
        all_preds = np.full((n_cells, n_targets), np.nan)
        fold_r2s = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(range(n_cells))):
            # Z-score expression on train
            x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0)
            x_std[x_std == 0] = 1.0
            X_norm = (X - x_mean) / x_std

            # Z-score ephys on train
            y_mean, y_std = y[train_idx].mean(0), y[train_idx].std(0)
            y_std[y_std == 0] = 1.0
            y_norm = (y - y_mean) / y_std

            preds, fold_r2 = train_mlp(
                X_norm[train_idx], y_norm[train_idx],
                X_norm[test_idx], y_norm[test_idx],
                n_targets,
            )
            all_preds[test_idx] = preds
            fold_r2s.append(fold_r2)
            print(f"  Fold {fold_i+1}/{args.cv} R²={fold_r2:.4f}")

        # Global metrics (on z-scored labels from last fold — recompute properly)
        # For proper global R², we need predictions in the same scale
        # Since each fold has different z-scoring, compute per-fold then average
        results = {
            "dataset": args.dataset,
            "round": args.round,
            "model": "mlp_ic",
            "n_ic_genes": len(ic_genes),
            "n_cells": n_cells,
            "n_features": n_targets,
            "mean_fold_r2": float(np.mean(fold_r2s)),
            "std_fold_r2": float(np.std(fold_r2s)),
            "fold_r2s": [float(r) for r in fold_r2s],
        }
        with open(out_dir / f"cv{args.cv}_mlp_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMLP: Mean fold R²={results['mean_fold_r2']:.4f} ± {results['std_fold_r2']:.4f}")

    else:
        # LODO — similar structure, group by donor
        donors = [donor_map.get(sid, donor_map.get(str(sid), "unknown")) for sid in common]
        unique_donors = sorted(set(donors))
        fold_r2s = []

        for donor in unique_donors:
            test_idx = np.array([i for i, d in enumerate(donors) if d == donor])
            train_idx = np.array([i for i, d in enumerate(donors) if d != donor])
            if len(test_idx) < 2:
                continue

            x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0)
            x_std[x_std == 0] = 1.0
            X_norm = (X - x_mean) / x_std

            y_mean, y_std = y[train_idx].mean(0), y[train_idx].std(0)
            y_std[y_std == 0] = 1.0
            y_norm = (y - y_mean) / y_std

            preds, fold_r2 = train_mlp(
                X_norm[train_idx], y_norm[train_idx],
                X_norm[test_idx], y_norm[test_idx],
                n_targets,
            )
            fold_r2s.append({"donor": donor, "r2": float(fold_r2), "n_cells": len(test_idx)})

        results = {
            "dataset": args.dataset,
            "model": "mlp_ic",
            "cv": "lodo",
            "donor_r2s": fold_r2s,
            "mean_r2": float(np.mean([f["r2"] for f in fold_r2s])),
        }
        with open(out_dir / "lodo_mlp_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMLP LODO: Mean R²={results['mean_r2']:.4f}")


if __name__ == "__main__":
    main()
```

**Step 2: Run MLP baselines (same experiments as GF)**

```bash
conda run -n geneformer python PatchSeq/scripts/05_mlp_baseline.py --dataset gaba --round 1 --cv 10
conda run -n geneformer python PatchSeq/scripts/05_mlp_baseline.py --dataset gaba --round 1 --cv 0
conda run -n geneformer python PatchSeq/scripts/05_mlp_baseline.py --dataset excitatory --round 1 --cv 10
conda run -n geneformer python PatchSeq/scripts/05_mlp_baseline.py --dataset pooled --round 1 --cv 10
```

**Step 3: Commit**

```bash
git add PatchSeq/scripts/05_mlp_baseline.py
git commit -m "Add MLP ion channel baseline for ephys prediction"
```

---

## Task 6: Analyze Results and Generate Figures

**Files:**
- Create: `PatchSeq/scripts/06_analyze_results.py`
- Output: `PatchSeq/results/` — PNG figures, summary tables

**Context:**
- Load all JSON result files from Task 4 and 5
- Generate comparison tables and plots matching the mouse results format
- Key comparisons: GF vs MLP, GABA vs Excitatory, 10-fold vs LODO, Round 1 vs Round 2

**Step 1: Write analysis script**

```python
"""06_analyze_results.py — Compare GF vs MLP results, generate figures."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/results")

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

# ── Load all results ───────────────────────────────────────────────
datasets = ["gaba", "excitatory", "pooled"]
results = {}
for ds in datasets:
    for rd in [1, 2]:
        base = RESULTS_DIR / ds / f"round{rd}"
        for cv_type in ["cv10", "lodo"]:
            for model in ["gf", "mlp"]:
                key = f"{ds}_r{rd}_{cv_type}_{model}"
                path = base / f"{cv_type}_{model}_results.json"
                results[key] = load_json(path)

# ── Summary table ──────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"{'Dataset':<15} {'Round':<7} {'CV':<7} {'GF R²':<12} {'MLP R²':<12} {'Delta':<10}")
print(f"{'='*80}")

for ds in datasets:
    for rd in [1, 2]:
        for cv in ["cv10", "lodo"]:
            gf = results.get(f"{ds}_r{rd}_{cv}_gf")
            mlp = results.get(f"{ds}_r{rd}_{cv}_mlp")
            if gf is None and mlp is None:
                continue
            gf_r2 = gf.get("global_r2", gf.get("mean_fold_r2", float("nan"))) if gf else float("nan")
            mlp_r2 = mlp.get("mean_fold_r2", float("nan")) if mlp else float("nan")
            delta = gf_r2 - mlp_r2 if np.isfinite(gf_r2) and np.isfinite(mlp_r2) else float("nan")
            print(f"{ds:<15} {rd:<7} {cv:<7} {gf_r2:<12.4f} {mlp_r2:<12.4f} {delta:<+10.4f}")

# ── Per-feature R² comparison (GABA Round 1, 10-fold) ─────────────
gf_gaba = results.get("gaba_r1_cv10_gf")
if gf_gaba and "per_feature_r2" in gf_gaba:
    feat_r2 = gf_gaba["per_feature_r2"]
    features = list(feat_r2.keys())
    values = list(feat_r2.values())

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    sorted_idx = np.argsort(values)[::-1]
    ax.barh(range(len(features)), [values[i] for i in sorted_idx], color="#3498DB", alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("R²")
    ax.set_title("Per-feature R² — GF-V1 on GABA (10-fold CV)")
    ax.invert_yaxis()

    fig.savefig(RESULTS_DIR / "gaba_per_feature_r2.png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"\nSaved: {RESULTS_DIR}/gaba_per_feature_r2.png")

# ── GF vs MLP bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

comparisons = []
for ds in ["gaba", "excitatory", "pooled"]:
    gf = results.get(f"{ds}_r1_cv10_gf")
    mlp = results.get(f"{ds}_r1_cv10_mlp")
    if gf and mlp:
        gf_r2 = gf.get("global_r2", gf.get("mean_fold_r2", 0))
        mlp_r2 = mlp.get("mean_fold_r2", 0)
        comparisons.append((ds, gf_r2, mlp_r2))

if comparisons:
    x = np.arange(len(comparisons))
    w = 0.35
    ax.bar(x - w/2, [c[1] for c in comparisons], w, label="GF-V1 (top2)", color="#E74C3C", alpha=0.8)
    ax.bar(x + w/2, [c[2] for c in comparisons], w, label="MLP (IC genes)", color="#3498DB", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0].capitalize() for c in comparisons])
    ax.set_ylabel("R² (10-fold CV)")
    ax.set_title("Ephys Prediction: Geneformer V1 vs MLP Baseline")
    ax.legend()

    fig.savefig(RESULTS_DIR / "gf_vs_mlp_comparison.png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR}/gf_vs_mlp_comparison.png")

# ── Mouse vs Human comparison table ───────────────────────────────
print(f"\n{'='*80}")
print("CROSS-SPECIES COMPARISON (10-fold CV, top2 freeze)")
print(f"{'='*80}")
print(f"{'Dataset':<25} {'Species':<10} {'Cells':<8} {'Features':<10} {'GF R²':<10} {'MLP R²':<10}")
print(f"{'─'*80}")
# Mouse results (from design doc)
print(f"{'M1 (mixed cortical)':<25} {'Mouse':<10} {'1,033':<8} {'22':<10} {'0.435':<10} {'0.365':<10}")
print(f"{'V1 (mixed cortical)':<25} {'Mouse':<10} {'4,107':<8} {'22':<10} {'0.345':<10} {'0.295':<10}")
# Human results
for ds in ["gaba", "excitatory", "pooled"]:
    gf = results.get(f"{ds}_r1_cv10_gf")
    mlp = results.get(f"{ds}_r1_cv10_mlp")
    if gf:
        n = gf.get("n_cells", "?")
        nf = gf.get("n_features", "?")
        gf_r2 = f"{gf.get('global_r2', gf.get('mean_fold_r2', 0)):.3f}"
        mlp_r2 = f"{mlp.get('mean_fold_r2', 0):.3f}" if mlp else "—"
        label = {"gaba": "GABA interneurons", "excitatory": "L2/3 excitatory", "pooled": "Pooled (all)"}[ds]
        print(f"{label:<25} {'Human':<10} {str(n):<8} {str(nf):<10} {gf_r2:<10} {mlp_r2:<10}")

print(f"\nAll results in: {RESULTS_DIR}/")
```

**Step 2: Run analysis**

```bash
conda run -n geneformer python PatchSeq/scripts/06_analyze_results.py
```

**Step 3: Commit**

```bash
git add PatchSeq/scripts/06_analyze_results.py
git commit -m "Add results analysis and cross-species comparison"
```

---

## Execution Order Summary

```
Task 1: Extract RData → h5ad (requires R, interactive inspection)
   ↓
Task 2: Tokenize with GF-V1
   ↓
Task 3: Prepare ephys features
   ↓
Task 4 + Task 5 (can run in parallel per dataset):
   ├── 04: GF fine-tuning (GABA r1 → GABA r2 → Exc → Pooled)
   └── 05: MLP baseline (same order)
   ↓
Task 6: Analyze and compare results
```

Tasks 1-3 are sequential (each depends on previous). Tasks 4-5 are independent per dataset and can be parallelized. Task 6 runs after all experiments complete.
