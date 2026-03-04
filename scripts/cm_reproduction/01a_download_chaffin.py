# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Download and Inspect Chaffin et al. 2022 Cardiomyopathy Data
#
# **Paper**: "Single-nucleus profiling of human dilated and hypertrophic
# cardiomyopathy" (Chaffin et al., Nature 608, 174-180, 2022)
#
# **Data sources**:
# - Broad Single Cell Portal: SCP1303 (requires web registration)
# - CZ CELLxGENE Census: NOT available (checked 2026-03-03)
# - HuggingFace pre-tokenized (V1): `ctheodoris/Genecorpus-30M` -> available locally
#
# **Conclusion**: The raw h5ad is not publicly available via programmatic API.
# The Broad SCP1303 requires manual web download. We use the V1 pre-tokenized
# dataset as reference for metadata/schema, but will need to re-tokenize with
# V2 settings if raw data is obtained.

# %%
from pathlib import Path

# Paths
DATA_ROOT = Path("/home/jw3514/Work/Geneformer/Geneformer/data")
V1_DATASET_PATH = (
    DATA_ROOT
    / "Genecorpus-30M"
    / "example_input_files"
    / "cell_classification"
    / "disease_classification"
    / "human_dcm_hcm_nf.dataset"
)
CHAFFIN_OUTPUT_DIR = DATA_ROOT / "chaffin_cardiomyopathy"
CHAFFIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1: Check for locally available data

# %%
print("=" * 60)
print("Step 1: Checking for locally available data")
print("=" * 60)

# Check V1 pre-tokenized dataset
if V1_DATASET_PATH.exists():
    print(f"[OK] V1 pre-tokenized dataset found: {V1_DATASET_PATH}")
    # Check size
    total_size = sum(f.stat().st_size for f in V1_DATASET_PATH.rglob("*") if f.is_file())
    print(f"     Total size: {total_size / 1e9:.2f} GB")
else:
    print(f"[MISSING] V1 pre-tokenized dataset not found at {V1_DATASET_PATH}")

# Check for raw h5ad
raw_h5ad = list(CHAFFIN_OUTPUT_DIR.glob("*.h5ad"))
if raw_h5ad:
    print(f"[OK] Raw h5ad found: {raw_h5ad}")
else:
    print(f"[MISSING] No raw h5ad in {CHAFFIN_OUTPUT_DIR}")

# %% [markdown]
# ## Step 2: Attempt CELLxGENE Census download
#
# The Chaffin 2022 dataset is NOT in CELLxGENE Census (verified 2026-03-03).
# The Census contains:
# - DCM: 934K cells from "Pathogenic variants damage cell composition..."
#   (different study, not Chaffin)
# - HCM: only 1,158 cells from "High-resolution single-cell transcriptomic
#   survey of cardiomyocytes from patients with HCM" (different study)
# - Normal heart: 4.3M cells from various atlases
#
# None of these correspond to the Chaffin 2022 cohort with matched NF/HCM/DCM.

# %%
print("=" * 60)
print("Step 2: CELLxGENE Census check (informational)")
print("=" * 60)

try:
    import cellxgene_census

    print(f"cellxgene_census version: {cellxgene_census.__version__}")
    print(
        "NOTE: Chaffin 2022 data is NOT in CELLxGENE Census."
    )
    print(
        "      The Census has DCM/HCM from different studies with"
        " different cohorts."
    )
    print(
        "      We need the original Chaffin data for matched"
        " NF/HCM/DCM comparison."
    )
except ImportError:
    print("cellxgene_census not installed (not needed - data not in Census)")

# %% [markdown]
# ## Step 3: Inspect V1 pre-tokenized dataset
#
# Even though this is V1-tokenized (max 2048 tokens, ~25K gene vocabulary),
# it tells us the metadata schema and cell composition we need.

# %%
from collections import Counter

from datasets import load_from_disk

print("=" * 60)
print("Step 3: Inspecting V1 pre-tokenized dataset")
print("=" * 60)

ds = load_from_disk(str(V1_DATASET_PATH))

print(f"\nDataset type: {type(ds).__name__}")
print(f"Number of cells: {len(ds):,}")
print(f"Columns: {ds.column_names}")
print(f"\nFeatures:")
for name, feat in ds.features.items():
    print(f"  {name}: {feat}")

# %%
print("\n--- Sample row (first cell) ---")
sample = ds[0]
for k, v in sample.items():
    if k == "input_ids":
        print(f"  input_ids: [{v[0]}, {v[1]}, {v[2]}, ...] (len={len(v)})")
    else:
        print(f"  {k}: {v}")

# %%
print("\n--- Metadata distributions ---")
for col in ds.column_names:
    if col not in ["input_ids", "length", "attention_mask"]:
        vals = Counter(ds[col])
        print(f"\n{col} ({len(vals)} unique values):")
        for k, v in vals.most_common(25):
            print(f"  {k}: {v:,}")

# %%
import statistics

lengths = ds["length"]
print("\n--- Input length statistics ---")
print(f"  Min:    {min(lengths):,}")
print(f"  Max:    {max(lengths):,}")
print(f"  Mean:   {statistics.mean(lengths):,.0f}")
print(f"  Median: {statistics.median(lengths):,.0f}")
print(f"  Stdev:  {statistics.stdev(lengths):,.0f}")

# %%
print("\n--- V1 vs V2 token compatibility ---")
max_token_sample = max(max(ids) for ids in ds[:1000]["input_ids"])
print(f"  Max token ID (first 1000 cells): {max_token_sample}")
print(f"  V1 vocabulary: ~25,400 genes (gc30M)")
print(f"  V2 vocabulary: ~20,275 genes (gc104M)")
print(f"  => V1 tokens are NOT compatible with V2 models")
print(f"  => Raw data must be re-tokenized for V2-316M")

# %% [markdown]
# ## Step 4: Disease x cell type cross-tabulation

# %%
import pandas as pd

print("\n--- Disease x Cell Type cross-tabulation ---")
df = pd.DataFrame({"cell_type": ds["cell_type"], "disease": ds["disease"]})
ct = pd.crosstab(df["cell_type"], df["disease"], margins=True)
print(ct.to_string())

# %%
# Cardiomyocyte-specific breakdown
cm_types = ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]
cm_df = df[df["cell_type"].isin(cm_types)]
print(f"\n--- Cardiomyocytes only ({len(cm_df):,} cells) ---")
cm_ct = pd.crosstab(cm_df["cell_type"], cm_df["disease"], margins=True)
print(cm_ct.to_string())

# %% [markdown]
# ## Step 5: Individual/donor breakdown

# %%
print("\n--- Individuals per disease ---")
ind_df = pd.DataFrame(
    {"individual": ds["individual"], "disease": ds["disease"]}
)
for disease in ["nf", "hcm", "dcm"]:
    inds = ind_df[ind_df["disease"] == disease]["individual"].unique()
    print(f"  {disease.upper()}: {len(inds)} donors - {sorted(inds)}")

# %%
# Train/eval/test split from the paper/example
train_ids = [
    "1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678",
    "1561", "1304", "1610", "1430", "1472", "1707", "1726", "1504",
    "1425", "1617", "1631", "1735", "1582", "1722", "1622", "1630",
    "1290", "1479", "1371", "1549", "1515",
]
eval_ids = ["1422", "1510", "1539", "1606", "1702"]
test_ids = ["1437", "1516", "1602", "1685", "1718"]

print("\n--- Pre-defined train/eval/test split (from Geneformer examples) ---")
print(f"  Train: {len(train_ids)} donors")
print(f"  Eval:  {len(eval_ids)} donors")
print(f"  Test:  {len(test_ids)} donors")

# Count cells per split
all_individuals = set(ds["individual"])
known_ids = set(train_ids + eval_ids + test_ids)
missing = known_ids - all_individuals
extra = all_individuals - known_ids
if missing:
    print(f"  WARNING: IDs in splits but not in data: {missing}")
if extra:
    print(f"  WARNING: IDs in data but not in splits: {extra}")

# %% [markdown]
# ## Summary and Next Steps
#
# ### Data status
# - **V1 pre-tokenized dataset**: Available locally (579,159 cells, V1 tokens)
# - **Raw h5ad from Chaffin 2022**: NOT available programmatically
#   - Broad SCP1303 requires manual web download with registration
#   - CELLxGENE Census does not contain this specific dataset
#
# ### For V2-316M reproduction
# **Option A** (preferred): Download raw data from Broad SCP1303 manually,
# then re-tokenize with `TranscriptomeTokenizer` using V2 settings (4096 input,
# gc104M dictionaries).
#
# **Option B** (fallback): Use V1 pre-tokenized data with V1-10M model as
# baseline comparison. The V1 data has max_input_size=2048 and uses gc30M
# tokens, so it CANNOT be used directly with V2-316M.
#
# **Option C**: Try to download from Broad SCP1303 using their API
# (requires authentication token from https://singlecell.broadinstitute.org).
#
# ### Key metadata columns for disease classification
# - `disease`: "nf", "hcm", "dcm" (the classification target)
# - `cell_type`: filter to Cardiomyocyte1/2/3 for the paper's experiments
# - `individual`: donor ID (used for train/test splitting to avoid data leakage)
# - `sex`, `age`, `lvef`: additional covariates

# %%
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Dataset: Chaffin et al. 2022 - human DCM/HCM/NF")
print(f"V1 pre-tokenized: {V1_DATASET_PATH}")
print(f"  Total cells: {len(ds):,}")
print(f"  Cardiomyocytes: {len(cm_df):,}")
print(f"  Disease classes: nf ({Counter(ds['disease'])['nf']:,}), "
      f"hcm ({Counter(ds['disease'])['hcm']:,}), "
      f"dcm ({Counter(ds['disease'])['dcm']:,})")
print(f"  Donors: {len(all_individuals)}")
print(f"  Max input length: {max(lengths)} (V1 limit: 2048)")
print(f"  Token vocabulary: V1 (gc30M, ~25K genes)")
print(f"\nFor V2-316M: raw data re-tokenization required")
print(f"  Source: Broad SCP1303 (manual download)")
print(f"  Target: {CHAFFIN_OUTPUT_DIR}")
