# Geneformer Cardiomyopathy Reproduction Design

## Goal

Reproduce the cardiomyopathy experiments from the Geneformer V1 paper (Theodoris et al., Nature 2023) using the V2-316M model, then extract per-ion-channel expression changes from TF perturbations for future openCARP integration.

## Context

Our lab uses openCARP (biophysical heart simulator) to simulate normal/disordered hearts by changing ion conductance of individual cardiomyocyte models. We currently study single-gene KCNQ1. We want to use Geneformer to identify TFs with large effects on cardiomyocyte state, extract their downstream impact on ion channel genes, and feed those expression changes into openCARP as conductance scaling factors.

## Architecture

```
Phase 1: Data Acquisition & Tokenization
  Raw scRNA-seq (Chaffin et al. + fetal cardiomyocytes)
    -> TranscriptomeTokenizer (V2, 4096 input)
    -> Tokenized HuggingFace Datasets

Phase 2: Geneformer Analysis
  2A: Zero-shot in silico deletion (fetal cardiomyocytes, pretrained)
  2B: Disease classification (fine-tune NF vs HCM vs DCM)
  2C: In silico treatment analysis (fine-tuned, TF deletions)
  2D: Per-gene expression impact extraction (for ion channel genes)
```

openCARP bridge will be designed in a follow-up session.

---

## Phase 1: Data Acquisition & Tokenization

### 1.1 Download datasets

**Dataset A: Chaffin et al. 2022 (adult cardiomyocytes)**
- Paper: "Single-nucleus profiling of human dilated and hypertrophic cardiomyopathy" (Nature 608, 174-180)
- Source: GEO or Broad Single Cell Portal
- Required metadata columns: `disease` (nf/hcm/dcm), `cell_type` (Cardiomyocyte1/2/3), `individual` (patient IDs)
- Patients: non-failing (n=9), hypertrophic (n=11), dilated (n=9)
- Total cells: ~93,000 (after filtering for cardiomyocytes)

**Dataset B: Fetal cardiomyocytes**
- Reference 23 in paper (likely Asp et al. or similar fetal heart atlas)
- Used for zero-shot in silico deletion analysis
- Required: Ensembl gene IDs, count matrix, cell type annotations

### 1.2 Tokenize with V2 settings

```python
from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer(
    model_version="V2",
    model_input_size=4096,
    special_token=True,
    nproc=10,
)
tk.tokenize_data(
    data_directory="data/chaffin_cardiomyopathy/",
    output_directory="data/tokenized/",
    output_prefix="chaffin_cm",
    file_format="h5ad",
)
```

### 1.3 Validate tokenized data
- Check cell counts match expected (~93K cardiomyocytes for Chaffin)
- Verify metadata columns preserved (disease, cell_type, individual)
- Spot-check token distributions

---

## Phase 2: Geneformer Analysis

### 2A: Zero-shot in silico deletion (fetal cardiomyocytes)

**Purpose:** Replicate Fig. 2d from the paper — show that deleting known cardiomyopathy genes has a larger effect on cardiomyocyte embeddings than control genes.

**Method:**
- Use pretrained V2-316M (no fine-tuning)
- Define gene sets:
  - Disease genes: known cardiomyopathy + structural heart disease genes
  - Control genes: hyperlipidaemia genes (expressed in cardiomyocytes but affect other cell types)
- Delete each gene in silico, measure cosine similarity shift

```python
from geneformer import InSilicoPerturber, InSilicoPerturberStats

isp = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=cardiomyopathy_gene_list,  # Ensembl IDs
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data={"cell_type": ["cardiomyocyte"]},  # adjust to actual labels
    max_ncells=2000,
    forward_batch_size=200,
    model_version="V2",
    nproc=10,
)

isp.perturb_data(
    "models/Geneformer/Geneformer-V2-316M",
    "data/tokenized/fetal_cardiomyocytes.dataset",
    "outputs/zero_shot_deletion/",
    "fetal_cm_deletion",
)
```

**Stats:** Use `mode="vs_null"` to compare disease gene shifts vs control gene shifts.

**Expected output:** Box plots showing disease genes have significantly larger cosine similarity shifts (more deleterious effect) than control genes, replicating Fig. 2d.

### 2B: Disease classification (fine-tune on adult cardiomyocytes)

**Purpose:** Replicate Fig. 6a-b — fine-tune Geneformer to distinguish NF vs HCM vs DCM cardiomyocytes.

**Method:**
- Fine-tune V2-316M classifier on Chaffin et al. cardiomyocytes
- Patient-level split to avoid data leakage
- Target: 90% out-of-sample accuracy (paper result)

```python
from geneformer import Classifier

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": "all"},
    filter_data={"cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]},
    training_args={
        "num_train_epochs": 1,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 12,
        "warmup_steps": 500,
        "weight_decay": 0.01,
    },
    freeze_layers=4,  # freeze more layers for V2-316M (12 layers total)
    num_crossval_splits=1,
    forward_batch_size=200,
    model_version="V2",
    nproc=10,
    ngpu=1,
)
```

**Evaluation:** Accuracy, macro F1, confusion matrix. Plot with `cc.plot_conf_mat()` and `cc.plot_predictions()`.

### 2C: In silico treatment analysis (find therapeutic TF targets)

**Purpose:** Replicate Fig. 6c-e — identify genes whose deletion shifts diseased cardiomyocytes toward non-failing state.

**Method:**
1. Extract state embeddings (NF, HCM, DCM centroids) from fine-tuned model
2. Run in silico deletion of all genes (or TFs only) in diseased cardiomyocytes
3. Rank genes by shift toward non-failing state

**Two runs:**
- HCM -> NF: `start_state="hcm"`, `goal_state="nf"`, `alt_states=["dcm"]`
- DCM -> NF: `start_state="dcm"`, `goal_state="nf"`, `alt_states=["hcm"]`

```python
from geneformer import EmbExtractor, InSilicoPerturber, InSilicoPerturberStats

# Step 1: State embeddings
embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=3,
    filter_data=filter_data_dict,
    max_ncells=2000,
    emb_layer=-1,
    summary_stat="exact_mean",
    forward_batch_size=200,
    emb_mode="cell",
    model_version="V2",
    nproc=10,
)

state_embs = embex.get_state_embs(
    cell_states_to_model={"state_key": "disease", "start_state": "dcm",
                          "goal_state": "nf", "alt_states": ["hcm"]},
    model_directory=finetuned_model_dir,
    input_data_file=tokenized_data,
    output_directory=output_dir,
    output_prefix="state_embs",
)

# Step 2: In silico perturbation
isp = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb="all",
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cell",
    cell_states_to_model={...},
    state_embs_dict=state_embs,
    filter_data=filter_data_dict,
    max_ncells=2000,
    forward_batch_size=200,
    model_version="V2",
    nproc=10,
)

# Step 3: Stats
ispstats = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed="all",
    cell_states_to_model={...},
    model_version="V2",
)
```

**Expected output:** Ranked list of genes/TFs whose deletion most shifts diseased cardiomyocytes toward NF state. Paper found 447 genes for HCM, 478 for DCM, with 197 overlap.

### 2D: Per-gene expression impact extraction

**Purpose:** For the top TFs identified in 2C, extract how their deletion affects specific ion channel genes. This is the bridge to openCARP.

**Method:**
- Re-run perturbation for top ~50 TFs with `emb_mode="cell_and_gene"`
- This outputs per-gene embedding shifts for every gene in each cell
- Filter output for ion channel genes of interest:
  - KCNQ1 (IKs), SCN5A (INa), CACNA1C (ICaL), KCNH2/HERG (IKr)
  - KCNJ2 (IK1), RYR2, ATP2A2/SERCA2a, SLC8A1/NCX
- Quantify: cosine similarity shift per ion channel gene per TF deletion

```python
isp_genes = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=top_tf_ensembl_ids,  # top 50 TFs from 2C
    model_type="CellClassifier",
    num_classes=3,
    emb_mode="cell_and_gene",  # key: get per-gene shifts
    cell_emb_style="mean_pool",
    filter_data=filter_data_dict,
    cell_states_to_model={...},
    state_embs_dict=state_embs,
    max_ncells=500,
    forward_batch_size=100,
    model_version="V2",
    nproc=10,
)
```

**Output:** CSV with columns: TF_deleted, ion_channel_gene, cosine_shift, significance. This becomes the input for future openCARP conductance mapping.

---

## Key Ion Channel Genes for openCARP Mapping

| Gene | Protein | Current | openCARP Parameter |
|------|---------|---------|-------------------|
| KCNQ1 | Kv7.1 | IKs | g_Ks |
| SCN5A | Nav1.5 | INa | g_Na |
| CACNA1C | Cav1.2 | ICaL | g_CaL |
| KCNH2 | hERG | IKr | g_Kr |
| KCNJ2 | Kir2.1 | IK1 | g_K1 |
| RYR2 | RyR2 | Jrel | - |
| ATP2A2 | SERCA2a | Jup | - |
| SLC8A1 | NCX1 | INCX | g_NaCa |

---

## Output Structure

```
outputs/
  phase1_tokenization/
    chaffin_cm.dataset           # tokenized adult cardiomyocytes
    fetal_cm.dataset             # tokenized fetal cardiomyocytes
  phase2a_zero_shot_deletion/
    fetal_cm_deletion_stats.csv  # per-gene cosine shifts
    plots/                       # box plots disease vs control
  phase2b_classification/
    cm_classifier_*/             # fine-tuned model checkpoints
    confusion_matrix.png
    predictions.png
  phase2c_treatment_analysis/
    hcm_to_nf_stats.csv         # genes shifting HCM -> NF
    dcm_to_nf_stats.csv         # genes shifting DCM -> NF
    top_tfs_ranked.csv           # top TFs by effect size
  phase2d_per_gene_impact/
    tf_ion_channel_shifts.csv   # per TF x per ion channel shifts
```

---

## Dependencies & Resources

- **Model:** Geneformer-V2-316M (already available at `models/Geneformer/Geneformer-V2-316M`)
- **GPU:** 1 GPU for fine-tuning and inference (check with `nvidia-smi`)
- **CPU:** 10 cores for data processing
- **Conda env:** `geneformer`
- **Estimated runtime:** Classification ~1-2h, full ISP (all genes) ~8-24h per direction

## Risks & Mitigations

- **V2 vs V1 results may differ:** V2-316M is more powerful but results won't match paper exactly. This is expected and acceptable since we want best results, not exact reproduction.
- **Data format differences:** Chaffin et al. data may have different column names than the paper's example files. Will need to inspect and map metadata columns.
- **ISP runtime for "all" genes:** Running perturbation for all ~20K genes is slow. Start with TFs only (~1,500 genes) for initial results, then expand if needed.
- **Memory:** V2-316M is 3x larger than V1. May need to reduce batch size or use quantization if OOM.
