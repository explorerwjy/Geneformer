# Human Patch-seq Ephys Prediction with Geneformer

## Goal

Test whether human Geneformer can predict electrophysiology features from single-cell transcriptomes in human Patch-seq data, following the successful mouse brain-Geneformer ephys fine-tuning approach (R²=0.435 on M1, 0.345 on V1).

## Datasets

### A. GABAergic interneurons (Lee & Dalley, Science 2023)
- **Source**: `/home/jw3514/Work/NeurSim/human_patchseq_gaba/`
- **Cells**: 704 with ephys, 778 total
- **Cell types**: PVALB (403), SST (171), VIP (122), LAMP5/PAX6 (82)
- **Regions**: MTG, temporal, frontal, parietal cortex
- **Conditions**: acute (311) vs culture (467); epilepsy/tumor patients
- **Ephys**: 93 features (80 with >80% coverage)
- **Gene expression**: RData format, needs R extraction to h5ad
- **DANDI**: 000636

### B. L2/3 excitatory neurons (Berg et al., eLife 2021)
- **Source**: `/home/jw3514/Work/NeurSim/patchseq_human_L23/`
- **Cells**: 304 with RNA Pass + ephys
- **Cell types**: FREM3 (205), LTK (88), COL22A1 (42), GLP2R (30), CARM1P1 (20)
- **Regions**: temporal (329), frontal (43), parietal (9)
- **Ephys**: 18 features (all mappable to GABA equivalents)
- **Gene expression**: RData format, needs R extraction to h5ad
- **Donors**: 82

### Poolability Assessment

Excitatory and GABAergic neurons have systematically different ephys profiles (expected biology):

| Feature | Excitatory | GABAergic | Ratio |
|---------|-----------|-----------|-------|
| Firing rate (Hz) | 6.6 | 19.2 | 0.34x |
| f-I slope | 0.11 | 0.38 | 0.30x |
| Input resistance (MΩ) | 130 | 231 | 0.56x |
| Rheobase (pA) | 136 | 76 | 1.78x |
| Sag ratio | 0.07 | 0.23 | 0.30x |

Strategy: run each dataset independently first (Phase 1), then pool to test combined training (Phase 2).

## Model

### Primary: Geneformer V1-10M
- Architecture: 6L, 256d BERT (same family as mouse brain-GF)
- Cell embedding: **gene-level mean pool** (matching mouse approach)
- Token dictionary: V1 30M corpus (`gene_dictionaries_30m/`)
- Model path: `models/Geneformer/Geneformer-V1-10M/`
- Max input: 2048 tokens

### Regression head
```
[tokenized cell] → GF-V1 (6L, 256d) → mean_pool(gene embeddings)
                    → Dropout(0.1) → Linear(256, N_features)
                    → N ephys features
```

### Freeze strategy: top2
- Freeze: embeddings + layers 0-3
- Fine-tune: layers 4-5 + regression head (~600K / 9.5M params = 6.2%)
- Rationale: optimal on mouse data; prevents overfitting with <1K cells

### Comparison: MLP baseline on ion channel genes
```
[~140 IC genes, log2(CPM+1)] → Linear(140, 256) → Tanh
                               → Dropout(0.1) → Linear(256, N_features)
```
Human ion channel gene selection: SCN*, KCNA-KCNV*, CACNA*, HCN*, CLCN* + markers (PVALB, SST, VIP, LAMP5).

## Ephys Feature Sets

### Round 1: 18 matched features (shared between datasets)
All 18 Berg features map to GABA equivalents:

| Berg name | GABA equivalent | Description |
|-----------|----------------|-------------|
| adapt_mean | adapt_mean | Spike frequency adaptation |
| avg_rate_hero | avg_rate_hero | Mean firing rate |
| downstroke_long_square | downstroke_hero | AP downstroke rate |
| fast_trough_v_long_square_rel | fast_trough_deltav_hero | AP fast AHP |
| fi_fit_slope | fi_fit_slope | f-I curve slope |
| first_isi_hero_inv | first_isi_inv_hero | Initial firing rate |
| input_resistance | input_resistance | Input resistance |
| latency_rheo | latency_rheo | Spike latency |
| peak_v_long_square_rel | peak_deltav_hero | AP height |
| rheobase_i | rheobase_i | Rheobase |
| sag | sag | Sag ratio |
| tau | tau | Membrane time constant |
| threshold_v_long_square | threshold_v_hero | AP threshold |
| trough_v_long_square_rel | trough_deltav_hero | AP trough |
| upstroke_downstroke_ratio_long_square | upstroke_downstroke_ratio_hero | AP upstroke/downstroke ratio |
| upstroke_long_square | upstroke_hero | AP upstroke rate |
| v_baseline | v_baseline | Resting membrane potential |
| width_long_square | width_hero | AP width |

### Round 2: Extended GABA features (~80 features)
Expand to all GABA features with >80% data coverage. Drop cells with >20% missing. GABA-only analysis (Berg doesn't have these features).

## Evaluation

### CV strategies
1. **10-fold CV** (standard, matches mouse experiment)
2. **Leave-one-donor-out CV** (tests cross-patient generalization)

### Metrics
- **Global R²**: `r2_score(y_true, y_pred, multioutput="uniform_average")`
- **Per-feature R²**: individual R² for each ephys feature
- **Mean Pearson r**: average correlation across features
- **Per-fold R²**: fold-by-fold stability

### Preprocessing
- Ephys targets: z-score standardized per fold (fit on train, transform test)
- Gene expression: rank value encoding via Geneformer tokenizer (V1 dictionary)
- MLP baseline: log2(CPM+1) normalized expression, z-scored per fold

## Experiment Phases

### Phase 1: Independent datasets (A first, then C)

**Experiment 1A: GABA (704 cells, 18 features)**
- GF-V1 fine-tuned (top2) — 10-fold CV + leave-one-donor-out
- MLP baseline on IC genes — same evaluation

**Experiment 1B: GABA extended (704 cells, ~80 features)**
- GF-V1 fine-tuned (top2) — 10-fold CV + leave-one-donor-out
- MLP baseline on IC genes — same evaluation

**Experiment 1C: Excitatory (304 cells, 18 features)**
- GF-V1 fine-tuned (top2) — 10-fold CV + leave-one-donor-out
- MLP baseline on IC genes — same evaluation

### Phase 2: Pooled (1,008 cells, 18 shared features)
- Naive pooling: combine all cells, z-score together
- Compare to Phase 1 individual results
- Analyze: does combining excitatory + inhibitory help or hurt?

## Pipeline Scripts

All scripts in `PatchSeq/` within the Geneformer repo.

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_extract_rdata.R` | Load both RData files, merge, export gene expression as h5ad |
| 02 | `02_tokenize.py` | Tokenize with V1 dictionaries (max 2048 tokens) |
| 03 | `03_prepare_ephys.py` | Load ephys CSVs, select/map features, align with cells, handle missing values |
| 04 | `04_finetune_gf_ephys.py` | GF-V1 fine-tuning with regression head (top2), 10-fold + LODO CV |
| 05 | `05_mlp_baseline.py` | MLP on human IC genes, same evaluation protocol |
| 06 | `06_analyze_results.py` | Compare GF vs MLP, per-feature R², cross-dataset comparison, figures |

## Output Structure

```
PatchSeq/
├── scripts/
│   ├── 01_extract_rdata.R
│   ├── 02_tokenize.py
│   ├── 03_prepare_ephys.py
│   ├── 04_finetune_gf_ephys.py
│   ├── 05_mlp_baseline.py
│   └── 06_analyze_results.py
├── results/
│   ├── gaba/           # Experiment 1A + 1B
│   ├── excitatory/     # Experiment 1C
│   └── pooled/         # Phase 2
└── data/
    ├── gaba_expression.h5ad
    ├── excitatory_expression.h5ad
    ├── gaba_tokenized.dataset/
    └── excitatory_tokenized.dataset/
```

## Hardware

- GPU: 16GB VRAM sufficient for V1-10M (399MB model)
- Fine-tuning (fp16 + gradient checkpointing): ~4-6 GB VRAM
- CPU: 10 cores for data preprocessing

## Risks

- **Small sample size**: 704 (GABA) and 304 (excitatory) cells are small for deep learning. Top2 freeze is essential.
- **Gene ID mapping**: V1 token dictionary uses human Ensembl IDs from Genecorpus-30M. Need to verify coverage of genes detected in Patch-seq.
- **RData extraction**: Requires R environment; gene expression matrix format in RData is non-standard (cbind of two objects).
- **Excitatory dataset is very small**: 304 cells may be insufficient for stable GF fine-tuning. MLP baseline may be more competitive here.
- **Culture vs acute**: Culture conditions may alter both transcriptome and ephys. Not filtering — letting the model handle this — could add noise.
