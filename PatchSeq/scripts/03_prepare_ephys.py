"""
03_prepare_ephys.py

Load ephys CSVs from GABA and excitatory datasets, align feature names,
and save prepared pickle files for downstream modeling.

Outputs:
  - gaba_ephys_r1.pkl: 18 shared features (Berg canonical names)
  - gaba_ephys_r2.pkl: All GABA features with >80% coverage, cells with <=20% missing
  - excitatory_ephys.pkl: 18 features (Berg canonical names), RNA Pass cells only
"""

import pickle
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GABA_EPHYS = Path(
    "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/LeeDalley_ephys_fx.csv"
)
GABA_META = Path(
    "/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/LeeDalley_manuscript_metadata.csv"
)
EXC_EPHYS = Path(
    "/home/jw3514/Work/NeurSim/patchseq_human_L23/data/human_mouse_ephys_all_0127.csv"
)
EXC_META = Path(
    "/home/jw3514/Work/NeurSim/patchseq_human_L23/data/human_IVSCC_excitatory_L23_consolidated_0131.csv"
)

OUT_DIR = Path("/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature mapping: Berg long_square names (canonical) -> GABA hero equivalents
# Keys = Berg canonical names, Values = GABA column names
# ---------------------------------------------------------------------------
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

# Reverse: GABA name -> Berg canonical name
FEATURE_MAP_GABA_TO_BERG = {v: k for k, v in FEATURE_MAP_BERG_TO_GABA.items()}

# Canonical feature order (sorted Berg names)
CANONICAL_FEATURES = sorted(FEATURE_MAP_BERG_TO_GABA.keys())


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")


def load_gaba():
    """Load GABA ephys and metadata, build donor map."""
    ephys = pd.read_csv(GABA_EPHYS, index_col="specimen_id")
    meta = pd.read_csv(GABA_META)

    # Build donor map from metadata (specimen_id -> Donor)
    donor_map = dict(zip(meta["specimen_id"], meta["Donor"]))

    print(f"GABA ephys: {ephys.shape[0]} cells x {ephys.shape[1]} features")
    print(f"GABA metadata: {meta.shape[0]} rows, {len(donor_map)} unique specimen_ids")

    return ephys, donor_map


def load_excitatory():
    """Load excitatory ephys and metadata, filter to RNA Pass, build donor map."""
    ephys = pd.read_csv(EXC_EPHYS, index_col="specimen_id")
    meta = pd.read_csv(EXC_META)

    print(f"Excitatory ephys (raw): {ephys.shape[0]} cells x {ephys.shape[1]} features")
    print(f"Excitatory metadata: {meta.shape[0]} rows")

    # Filter to RNA Pass quality cells using metadata
    pass_ids = set(
        meta.loc[meta["rna_amplification_call"] == "Pass", "SpecimenID"]
    )
    ephys = ephys.loc[ephys.index.isin(pass_ids)]
    print(f"Excitatory ephys after RNA Pass filter: {ephys.shape[0]} cells")

    # Build donor map (SpecimenID -> donor) for cells in ephys
    meta_pass = meta[meta["SpecimenID"].isin(ephys.index)]
    donor_map = dict(zip(meta_pass["SpecimenID"], meta_pass["donor"]))

    return ephys, donor_map


def prepare_gaba_round1(ephys, donor_map):
    """
    Round 1: Select 18 shared features from GABA, rename to Berg canonical names.
    """
    gaba_cols = [FEATURE_MAP_BERG_TO_GABA[f] for f in CANONICAL_FEATURES]
    ephys_r1 = ephys[gaba_cols].copy()
    ephys_r1.columns = CANONICAL_FEATURES  # Rename to Berg canonical

    result = {
        "ephys": ephys_r1,
        "donor_map": {
            sid: donor_map[sid] for sid in ephys_r1.index if sid in donor_map
        },
        "feature_names": list(ephys_r1.columns),
    }

    print(f"\nGABA Round 1:")
    print(f"  Shape: {ephys_r1.shape}")
    print(f"  NaN count: {ephys_r1.isna().sum().sum()}")
    print(f"  Donor map entries: {len(result['donor_map'])}")

    return result


def prepare_gaba_round2(ephys, donor_map):
    """
    Round 2: All GABA features with >80% non-null coverage.
    Drop cells with >20% missing values across selected features, then dropna.
    """
    # Select features with >80% coverage
    coverage = ephys.notna().mean()
    good_features = coverage[coverage > 0.80].index.tolist()
    ephys_r2 = ephys[good_features].copy()
    print(f"\nGABA Round 2:")
    print(f"  Features with >80% coverage: {len(good_features)} (from {ephys.shape[1]})")

    # Drop cells with >20% missing values across selected features
    cell_missing_frac = ephys_r2.isna().mean(axis=1)
    n_before = ephys_r2.shape[0]
    ephys_r2 = ephys_r2[cell_missing_frac <= 0.20]
    print(f"  Cells after dropping >20% missing: {ephys_r2.shape[0]} (dropped {n_before - ephys_r2.shape[0]})")

    # Final dropna
    n_before2 = ephys_r2.shape[0]
    ephys_r2 = ephys_r2.dropna()
    print(f"  Cells after dropna: {ephys_r2.shape[0]} (dropped {n_before2 - ephys_r2.shape[0]})")

    result = {
        "ephys": ephys_r2,
        "donor_map": {
            sid: donor_map[sid] for sid in ephys_r2.index if sid in donor_map
        },
        "feature_names": list(ephys_r2.columns),
    }

    print(f"  Final shape: {ephys_r2.shape}")
    print(f"  Donor map entries: {len(result['donor_map'])}")

    return result


def prepare_excitatory(ephys, donor_map):
    """
    Excitatory: 18 features already use Berg canonical names.
    """
    # Verify the columns match our canonical features
    exc_features = sorted(ephys.columns.tolist())
    assert exc_features == CANONICAL_FEATURES, (
        f"Excitatory features mismatch.\n"
        f"  Expected: {CANONICAL_FEATURES}\n"
        f"  Got: {exc_features}"
    )

    ephys_exc = ephys[CANONICAL_FEATURES].copy()

    result = {
        "ephys": ephys_exc,
        "donor_map": donor_map,
        "feature_names": list(ephys_exc.columns),
    }

    print(f"\nExcitatory:")
    print(f"  Shape: {ephys_exc.shape}")
    print(f"  NaN count: {ephys_exc.isna().sum().sum()}")
    print(f"  Donor map entries: {len(result['donor_map'])}")

    return result


def verify_outputs():
    """Load and verify all saved pickle files."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    for fname in ["gaba_ephys_r1.pkl", "gaba_ephys_r2.pkl", "excitatory_ephys.pkl"]:
        path = OUT_DIR / fname
        with open(path, "rb") as f:
            data = pickle.load(f)

        ephys = data["ephys"]
        print(f"\n{fname}:")
        print(f"  Keys: {list(data.keys())}")
        print(f"  Ephys shape: {ephys.shape}")
        print(f"  Ephys index dtype: {ephys.index.dtype}")
        print(f"  Ephys index name: {ephys.index.name}")
        print(f"  Ephys columns: {list(ephys.columns)}")
        print(f"  NaN total: {ephys.isna().sum().sum()}")
        print(f"  NaN per column:\n{ephys.isna().sum().to_string()}")
        print(f"  Donor map size: {len(data['donor_map'])}")
        print(f"  Feature names: {data['feature_names']}")
        print(f"  Sample index values: {list(ephys.index[:3])}")

        # Check donor map specimen_id type matches index type
        if data["donor_map"]:
            sample_key = next(iter(data["donor_map"]))
            print(f"  Donor map key type: {type(sample_key).__name__}")
            print(f"  Index dtype: {ephys.index.dtype}")


def main():
    print("=" * 60)
    print("Preparing Ephys Features")
    print("=" * 60)

    # Load data
    gaba_ephys, gaba_donors = load_gaba()
    exc_ephys, exc_donors = load_excitatory()

    # Prepare and save
    gaba_r1 = prepare_gaba_round1(gaba_ephys, gaba_donors)
    save_pickle(gaba_r1, OUT_DIR / "gaba_ephys_r1.pkl")

    gaba_r2 = prepare_gaba_round2(gaba_ephys, gaba_donors)
    save_pickle(gaba_r2, OUT_DIR / "gaba_ephys_r2.pkl")

    exc = prepare_excitatory(exc_ephys, exc_donors)
    save_pickle(exc, OUT_DIR / "excitatory_ephys.pkl")

    # Verify
    verify_outputs()

    print("\nDone.")


if __name__ == "__main__":
    main()
