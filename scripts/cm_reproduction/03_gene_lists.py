# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Curate gene lists for cardiomyopathy in silico perturbation
#
# Creates gene lists (Ensembl IDs) for:
# 1. **cardiomyopathy_genes** - Known cardiomyopathy genes (sarcomere, desmosome, lamin, etc.)
# 2. **structural_heart_genes** - GWAS-linked structural heart disease genes
# 3. **hyperlipidaemia_control** - Hyperlipidaemia genes (control set)
# 4. **ion_channel_genes** - Ion channel genes for openCARP mapping
# 5. **cardiac_tfs** - Cardiac-relevant transcription factors
# 6. **disease_combined** - Union of cardiomyopathy + structural heart genes

# %%
import pickle
from pathlib import Path

import pandas as pd
from geneformer import ENSEMBL_DICTIONARY_FILE

# %%
# Load gene name -> Ensembl ID mapping from Geneformer V2
with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
    gene_name_to_id = pickle.load(f)

print(f"Loaded {len(gene_name_to_id):,} gene name -> Ensembl ID mappings")

# %%
# Define gene lists by category
# Alias map for genes whose official symbol differs in the Geneformer dictionary
ALIASES = {
    "TAZ": "TAFAZZIN",  # TAZ (tafazzin) is listed as TAFAZZIN in V2 dict
}

gene_lists = {
    "cardiomyopathy_genes": [
        "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "MYL2", "MYL3", "ACTC1",
        "TTN", "TCAP", "LDB3", "CSRP3", "ACTN2", "DSP", "PKP2", "DSG2",
        "DSC2", "JUP", "LMNA", "SCN5A", "RYR2", "PLN", "CASQ2", "EMD",
        "TMEM43", "TBX5", "GATA4", "NKX2-5", "HAND2", "MEF2C", "FOXM1",
        "BAG3", "FLNC", "RBM20", "TNNC1", "DES", "VCL", "TAZ", "DTNA",
        "SGCD", "DMD", "LAMP2",
    ],
    "structural_heart_genes": [
        "ALPK3", "BAG3", "CDKN1A", "FLNC", "MTSS1", "PLN", "SMARCB1",
        "STRN", "SWI5", "SYNPO2L", "TBX20", "TNNT2", "TTN", "VCL",
    ],
    "hyperlipidaemia_control": [
        "LDLR", "PCSK9", "APOB", "APOE", "APOA1", "APOC3", "CETP", "LIPC",
        "LIPG", "LPL", "ABCA1", "ABCG1", "HMGCR", "NPC1L1", "SCARB1",
        "ANGPTL3", "ANGPTL4", "SORT1", "TRIB1", "GCKR",
    ],
    "ion_channel_genes": [
        "KCNQ1", "SCN5A", "CACNA1C", "KCNH2", "KCNJ2", "RYR2", "ATP2A2",
        "SLC8A1", "KCNA5", "KCNJ11", "CACNA1G", "HCN4",
    ],
    "cardiac_tfs": [
        "GATA4", "GATA6", "TBX5", "TBX20", "NKX2-5", "HAND1", "HAND2",
        "MEF2A", "MEF2C", "MEF2D", "TEAD1", "TEAD4", "SRF", "MYOCD",
        "FOXM1", "FOXO1", "FOXO3", "IRX4", "PITX2", "ISL1", "MEIS1",
        "MEIS2", "HEY1", "HEY2", "NOTCH1", "KLF2", "KLF4", "KLF15",
        "ETS1", "ETS2", "ERG", "STAT3", "NFAT5", "NFATC1", "NFATC2",
        "NFATC4", "PPARA", "PPARGC1A", "ESRRA", "ESRRG", "NR2F2", "TP53",
        "RB1", "MYC", "JUN", "FOS", "EGR1", "SMAD2", "SMAD3", "SMAD4",
        "CTCF", "YAP1", "WWTR1",
    ],
}


# %%
def resolve_gene_name(gene_name: str) -> str:
    """Resolve a gene name to the key used in the Geneformer dictionary."""
    if gene_name in gene_name_to_id:
        return gene_name
    if gene_name in ALIASES and ALIASES[gene_name] in gene_name_to_id:
        return ALIASES[gene_name]
    return gene_name  # will be flagged as missing


def convert_gene_list(
    gene_names: list[str],
    list_name: str,
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Convert gene names to Ensembl IDs.

    Returns (ensembl_ids, dataframe, missing_genes).
    """
    records = []
    ensembl_ids = []
    missing = []

    for gene in gene_names:
        resolved = resolve_gene_name(gene)
        ens_id = gene_name_to_id.get(resolved)
        if ens_id is not None:
            records.append(
                {
                    "gene_name": gene,
                    "dict_name": resolved,
                    "ensembl_id": ens_id,
                }
            )
            ensembl_ids.append(ens_id)
        else:
            missing.append(gene)

    df = pd.DataFrame(records)
    if missing:
        print(f"  WARNING [{list_name}]: {len(missing)} genes missing: {missing}")
    else:
        print(f"  [{list_name}]: all {len(gene_names)} genes mapped successfully")

    return ensembl_ids, df, missing


# %%
# Output directory
output_dir = Path("/home/jw3514/Work/Geneformer/Geneformer/data/gene_lists")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert and save each gene list
all_missing = {}
for name, genes in gene_lists.items():
    ensembl_ids, df, missing = convert_gene_list(genes, name)

    # Save pickle (list of Ensembl IDs for Geneformer)
    pkl_path = output_dir / f"{name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(ensembl_ids, f)

    # Save CSV (readable with gene names)
    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    print(f"  Saved {pkl_path.name} ({len(ensembl_ids)} genes) and {csv_path.name}")

    if missing:
        all_missing[name] = missing

# %%
# Create combined disease list (cardiomyopathy + structural heart)
cm_ids, _, _ = convert_gene_list(gene_lists["cardiomyopathy_genes"], "cm_tmp")
sh_ids, _, _ = convert_gene_list(gene_lists["structural_heart_genes"], "sh_tmp")

# Union (preserve order, deduplicate)
combined_ids = list(dict.fromkeys(cm_ids + sh_ids))

# Build combined DataFrame
id_to_name = {v: k for k, v in gene_name_to_id.items()}
combined_records = []
for ens_id in combined_ids:
    dict_name = id_to_name.get(ens_id, "UNKNOWN")
    # Find the original gene name from our lists
    original_name = dict_name
    for gene in gene_lists["cardiomyopathy_genes"] + gene_lists["structural_heart_genes"]:
        resolved = resolve_gene_name(gene)
        if gene_name_to_id.get(resolved) == ens_id:
            original_name = gene
            break
    combined_records.append(
        {
            "gene_name": original_name,
            "dict_name": dict_name,
            "ensembl_id": ens_id,
        }
    )

combined_df = pd.DataFrame(combined_records)

pkl_path = output_dir / "disease_combined.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(combined_ids, f)

csv_path = output_dir / "disease_combined.csv"
combined_df.to_csv(csv_path, index=False)

print(f"\n  [disease_combined]: {len(combined_ids)} unique genes (union of CM + structural)")
print(f"  Saved {pkl_path.name} and {csv_path.name}")

# %%
# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name in list(gene_lists.keys()) + ["disease_combined"]:
    pkl_path = output_dir / f"{name}.pkl"
    with open(pkl_path, "rb") as f:
        ids = pickle.load(f)
    print(f"  {name:30s}: {len(ids):3d} genes")

if all_missing:
    print("\nMISSING GENES (not in Geneformer V2 dictionary):")
    for name, genes in all_missing.items():
        print(f"  {name}: {genes}")
    print("\nNote: Aliases applied where possible (e.g., TAZ -> TAFAZZIN)")
else:
    print("\nAll genes successfully mapped to Ensembl IDs.")

print(f"\nOutput directory: {output_dir}")
