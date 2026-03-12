library(Seurat)
library(Matrix)

cat("Reading RDS file...\n")
obj <- readRDS("/home/jw3514/Work/Geneformer/data/QiuyanLiver/04_humnucseq_scrublet_dim30_celltype_doubletremoved_integrated.rds")

rna <- obj@assays[["RNA"]]

# Get features (genes) from the LogMap
features_lm <- attr(rna, "features")
cat("Features LogMap:\n")
cat("  class:", class(features_lm), "\n")
cat("  attr names:", names(attributes(features_lm)), "\n")
cat("  dim:", dim(features_lm), "\n")

# LogMap dimnames should have the gene names
feat_dimnames <- attr(features_lm, "dimnames")
cat("  dimnames class:", class(feat_dimnames), "\n")
cat("  dimnames length:", length(feat_dimnames), "\n")
if (is.list(feat_dimnames)) {
    for (i in seq_along(feat_dimnames)) {
        cat("  dimnames[[", i, "]] class:", class(feat_dimnames[[i]]),
            "len:", length(feat_dimnames[[i]]),
            "head:", head(feat_dimnames[[i]], 5), "\n")
    }
}

# Get cells from the LogMap
cells_lm <- attr(rna, "cells")
cat("\nCells LogMap:\n")
cat("  class:", class(cells_lm), "\n")
cat("  dim:", dim(cells_lm), "\n")
cells_dimnames <- attr(cells_lm, "dimnames")
if (is.list(cells_dimnames)) {
    for (i in seq_along(cells_dimnames)) {
        cat("  dimnames[[", i, "]] class:", class(cells_dimnames[[i]]),
            "len:", length(cells_dimnames[[i]]),
            "head:", head(cells_dimnames[[i]], 3), "\n")
    }
}

# The cell barcodes should also be available from metadata rownames
meta <- obj@meta.data
cells_from_meta <- rownames(meta)
cat("\nCells from metadata rownames:", length(cells_from_meta),
    "head:", head(cells_from_meta, 3), "\n")

# Get gene names from features LogMap rownames
genes <- feat_dimnames[[1]]
cells <- cells_from_meta

cat("\n=== Final dimensions ===\n")
cat("Genes:", length(genes), "\n")
cat("Cells:", length(cells), "\n")

# Extract counts
layers <- attr(rna, "layers")
counts <- layers[["counts"]]
cat("Counts dims:", dim(counts), "\n")

# Assign dimnames
rownames(counts) <- genes
colnames(counts) <- cells

# Save as MatrixMarket + genes + barcodes
cat("\nSaving counts.mtx...\n")
writeMM(counts, "/home/jw3514/Work/Geneformer/data/QiuyanLiver/counts.mtx")

cat("Saving genes.csv...\n")
write.csv(data.frame(gene = genes), "/home/jw3514/Work/Geneformer/data/QiuyanLiver/genes.csv", row.names = FALSE)

cat("Saving barcodes.csv...\n")
write.csv(data.frame(barcode = cells), "/home/jw3514/Work/Geneformer/data/QiuyanLiver/barcodes.csv", row.names = FALSE)

cat("Saving metadata.csv...\n")
write.csv(meta, "/home/jw3514/Work/Geneformer/data/QiuyanLiver/metadata.csv")

cat("\nDone!\n")
