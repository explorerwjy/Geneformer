# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geneformer is a foundational transformer model (BERT-based) pretrained on ~104M single-cell transcriptomes. It uses rank value encoding of genes to enable context-aware predictions for network biology tasks with limited data.

**Model variants**: V2-316M (default, 4096 input, ~20K genes), V2-104M, V2-104M_CLcancer, V1-10M (2048 input, ~25K genes).

## Environment & Installation

```bash
conda activate geneformer  # or check conda env list
pip install .               # setuptools-based, requires Python >=3.10
```

## Code Quality

Pre-commit hooks enforce **ruff** (linting + formatting), **isort** (black profile), trailing whitespace, and large file checks. Run manually:

```bash
pre-commit run --all-files
ruff check geneformer/
ruff format geneformer/
isort --profile black geneformer/
```

There is no formal test suite (no tests/ directory or pytest config).

## Architecture

### Core Data Flow

```
Raw scRNAseq (.loom/.h5ad/.zarr)
  → TranscriptomeTokenizer.tokenize_data()  [tokenizer.py]
  → Rank Value Encoding (genes ranked by median-scaled expression)
  → HuggingFace Dataset (input_ids, length, attention_mask)
  → Downstream task class (Classifier, EmbExtractor, InSilicoPerturber, etc.)
```

### Key Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `tokenizer.py` | `TranscriptomeTokenizer` | Converts raw expression → rank-encoded tokens |
| `classifier.py` | `Classifier` | Single-task cell/gene classification with CV and hyperparameter tuning |
| `mtl_classifier.py` + `mtl/` | `MTLClassifier`, `GeneformerMultiTask` | Multi-task learning with shared BERT encoder + task heads |
| `emb_extractor.py` | `EmbExtractor` | Extract cell/gene embeddings, UMAP visualization |
| `in_silico_perturber.py` | `InSilicoPerturber` | Simulate gene deletions/overexpressions |
| `in_silico_perturber_stats.py` | `InSilicoPerturberStats` | Statistical analysis of perturbation effects |
| `pretrainer.py` | `GeneformerPretrainer` | Pretraining with masked language modeling |
| `collator_for_classification.py` | `DataCollatorForCellClassification`, `DataCollatorForGeneClassification` | Batch preparation |
| `classifier_utils.py` | — | Downsampling, stratified splitting, label handling |
| `perturber_utils.py` | — | Shared utilities for perturbation analysis |

### Parameter Validation Pattern

All main classes use a `valid_option_dict` dictionary to validate constructor arguments:
```python
valid_option_dict = {
    "param_name": {allowed_types_or_values},
}
```

### Gene Dictionaries

Token dictionaries are pickle files bundled with the package (included via MANIFEST.in):
- V2 (104M corpus): `*_gc104M.pkl` files in `geneformer/`
- V1 (30M corpus): `geneformer/gene_dictionaries_30m/*_gc30M.pkl`

Constants for these are exported from `geneformer/__init__.py` (e.g., `GENE_MEDIAN_FILE`, `TOKEN_DICTIONARY_FILE`).

### Multi-Task Learning (mtl/)

The MTL module has its own training loop (`mtl/train.py`) with DDP support, mixed precision, and TensorBoard/WandB logging. Key files: `model.py` (shared encoder + heads), `data.py` (streaming dataset), `collators.py`, `utils.py` (optimizer/scheduler setup).

## Key Dependencies

- **torch + transformers (4.46) + peft**: Model backbone, fine-tuning (LoRA), training
- **anndata + scanpy + loompy**: Single-cell data I/O
- **datasets + pyarrow**: HuggingFace Dataset format for tokenized data
- **optuna + ray + hyperopt**: Hyperparameter optimization
- **tdigest**: Efficient percentile calculation for embedding summary stats

## Git LFS

Large files (.pkl, .pt, .bin, .safetensors, model dirs) are tracked with git-lfs. See `.gitattributes`.
