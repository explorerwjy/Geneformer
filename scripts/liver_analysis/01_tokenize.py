"""
Step 1: Tokenize the QiuyanLiver dataset for Geneformer.
Converts raw counts h5ad → rank value encoded HuggingFace Dataset.
"""

import sys

sys.path.insert(0, "/home/jw3514/Work/Geneformer/Geneformer")

from geneformer import TranscriptomeTokenizer

DATA_DIR = "/home/jw3514/Work/Geneformer/data/QiuyanLiver"
OUTPUT_DIR = "/home/jw3514/Work/Geneformer/data/QiuyanLiver/tokenized"

# Tokenize with V2 model settings, preserving metadata columns
tk = TranscriptomeTokenizer(
    custom_attr_name_dict={
        "Cell.type.new": "cell_type",
        "orig.ident": "sample_id",
        "nCount_RNA": "nCount_RNA_original",
        "nFeature_RNA": "nFeature_RNA",
        "percent.mt": "percent_mt",
    },
    nproc=10,
    model_version="V2",
)

tk.tokenize_data(
    data_directory=DATA_DIR,
    output_directory=OUTPUT_DIR,
    output_prefix="liver_qiuyan",
    file_format="h5ad",
)

print("\nTokenization complete!")
print(f"Output saved to: {OUTPUT_DIR}")
