"""
Tokenize GABA and excitatory h5ad files using Geneformer V1 tokenizer.

Uses input_identifier to filter files so we can tokenize each cell type
separately from the same data directory.
"""

import sys

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from geneformer import TranscriptomeTokenizer

DATA_DIR = "/home/jw3514/Work/Geneformer/Geneformer/PatchSeq/data"

tk = TranscriptomeTokenizer(
    custom_attr_name_dict={"specimen_id": "specimen_id"},
    nproc=10,
    model_input_size=2048,
    model_version="V1",
)

# Tokenize GABA cells
print("=" * 60)
print("Tokenizing GABA cells...")
print("=" * 60)
tk.tokenize_data(
    data_directory=DATA_DIR,
    output_directory=DATA_DIR,
    output_prefix="gaba_tokenized",
    file_format="h5ad",
    input_identifier="gaba_raw_counts",
)

# Tokenize excitatory cells
print("=" * 60)
print("Tokenizing excitatory cells...")
print("=" * 60)
tk.tokenize_data(
    data_directory=DATA_DIR,
    output_directory=DATA_DIR,
    output_prefix="excitatory_tokenized",
    file_format="h5ad",
    input_identifier="excitatory_raw_counts",
)

print("\nDone. Tokenized datasets saved to:")
print(f"  {DATA_DIR}/gaba_tokenized.dataset/")
print(f"  {DATA_DIR}/excitatory_tokenized.dataset/")
