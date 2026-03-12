#!/usr/bin/env Rscript
# 01_extract_rdata.R
# Extract gene expression data from RData files and export as CSV for h5ad conversion.
#
# Exports for each dataset:
#   - expression matrix (genes x cells) as gzipped CSV
#   - cell metadata as CSV
#   - gene names as CSV
#
# Usage: conda run -n geneformer Rscript PatchSeq/scripts/01_extract_rdata.R

library(Matrix)

out_dir <- "PatchSeq/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# GABA dataset (Lee & Dalley, Science 2023)
# ============================================================================
cat("=== Processing GABA dataset ===\n")

cat("Loading GABA RData files...\n")
load("/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/complete_patchseq_data_sets1.RData")
load("/home/jw3514/Work/NeurSim/human_patchseq_gaba/data/complete_patchseq_data_sets2.RData")

# Merge expression matrices
datPatch <- cbind(datPatch1, datPatch2)
cat(sprintf("  Merged expression matrix: %d genes x %d cells\n", nrow(datPatch), ncol(datPatch)))

# Verify alignment: annoPatch$sample_id matches colnames of datPatch
stopifnot(all(annoPatch$sample_id == colnames(datPatch)))
cat("  Cell order verified (annoPatch$sample_id == datPatch colnames)\n")

# Build cell metadata: specimen_id from metaPatch$spec_id, sample_id from annoPatch$sample_id
gaba_meta <- data.frame(
  sample_id = annoPatch$sample_id,
  specimen_id = as.integer(metaPatch$spec_id),
  donor = metaPatch$Donor,
  structure = metaPatch$structure,
  disease_state = metaPatch$disease_state,
  tree_first_cl = metaPatch$Tree_first_cl,
  subclass_label = metaPatch$subclass_label,
  broad_class_label = annoPatch$broad_class_label,
  genes_detected = annoPatch$genes_label,
  quality_score = annoPatch$quality_score_label,
  stringsAsFactors = FALSE
)
rownames(gaba_meta) <- gaba_meta$sample_id

cat(sprintf("  Metadata: %d cells, %d columns\n", nrow(gaba_meta), ncol(gaba_meta)))
cat(sprintf("  Specimen IDs: %d unique\n", length(unique(gaba_meta$specimen_id))))

# Export expression matrix (genes x cells) as gzipped CSV
cat("  Writing GABA expression matrix...\n")
gz_con <- gzfile(file.path(out_dir, "gaba_expression.csv.gz"), "w")
write.csv(datPatch, gz_con)
close(gz_con)

# Export metadata
cat("  Writing GABA metadata...\n")
write.csv(gaba_meta, file.path(out_dir, "gaba_cell_metadata.csv"), row.names = FALSE)

# Export gene names
cat("  Writing GABA gene names...\n")
write.csv(
  data.frame(gene_symbol = rownames(datPatch), stringsAsFactors = FALSE),
  file.path(out_dir, "gaba_gene_names.csv"),
  row.names = FALSE
)

cat("  GABA export complete.\n\n")

# Clean up GABA objects
rm(datPatch, datPatch1, datPatch2, annoPatch, metaPatch, gaba_meta)
gc()

# ============================================================================
# Excitatory dataset (Berg et al., eLife 2021) — human only
# ============================================================================
cat("=== Processing Excitatory dataset (human) ===\n")

cat("Loading excitatory RData file...\n")
load("/home/jw3514/Work/NeurSim/patchseq_human_L23/data/input_patchseq_data_sets.RData")

cat(sprintf("  Human expression matrix: %d genes x %d cells\n", nrow(datPatch), ncol(datPatch)))
cat(sprintf("  Mouse expression matrix: %d genes x %d cells (not exported)\n", nrow(datPatchM), ncol(datPatchM)))

# Verify alignment
stopifnot(all(annoPatch$sample_id == colnames(datPatch)))
cat("  Cell order verified (annoPatch$sample_id == datPatch colnames)\n")

# Build cell metadata
exc_meta <- data.frame(
  sample_id = annoPatch$sample_id,
  specimen_id = as.integer(annoPatch$SpecimenID),
  donor = annoPatch$donor,
  gender = annoPatch$gender,
  medical_conditions = annoPatch$medical_conditions,
  region = annoPatch$region,
  postPatch = annoPatch$postPatch,
  marker_sum_norm = annoPatch$marker_sum_norm,
  quality_score = annoPatch$quality_score,
  contaminationType = annoPatch$contaminationType,
  scaled_depth = annoPatch$scaled_depth,
  stringsAsFactors = FALSE
)
rownames(exc_meta) <- exc_meta$sample_id

cat(sprintf("  Metadata: %d cells, %d columns\n", nrow(exc_meta), ncol(exc_meta)))
cat(sprintf("  Specimen IDs: %d unique\n", length(unique(exc_meta$specimen_id))))

# Export expression matrix
cat("  Writing excitatory expression matrix...\n")
gz_con <- gzfile(file.path(out_dir, "excitatory_expression.csv.gz"), "w")
write.csv(datPatch, gz_con)
close(gz_con)

# Export metadata
cat("  Writing excitatory metadata...\n")
write.csv(exc_meta, file.path(out_dir, "excitatory_cell_metadata.csv"), row.names = FALSE)

# Export gene names
cat("  Writing excitatory gene names...\n")
write.csv(
  data.frame(gene_symbol = rownames(datPatch), stringsAsFactors = FALSE),
  file.path(out_dir, "excitatory_gene_names.csv"),
  row.names = FALSE
)

cat("  Excitatory export complete.\n\n")

cat("=== All exports done ===\n")
cat("Output files:\n")
for (f in list.files(out_dir, pattern = "\\.(csv|csv\\.gz)$")) {
  fpath <- file.path(out_dir, f)
  sz <- file.info(fpath)$size
  cat(sprintf("  %s  (%.1f MB)\n", f, sz / 1e6))
}
