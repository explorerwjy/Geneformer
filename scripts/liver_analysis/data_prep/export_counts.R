library(Seurat)
library(Matrix)

cat("Reading RDS file...\n")
obj <- readRDS("/home/jw3514/Work/Geneformer/data/QiuyanLiver/04_humnucseq_scrublet_dim30_celltype_doubletremoved_integrated.rds")

# Extract raw counts from RNA Assay5
rna <- obj@assays[["RNA"]]
layers <- attr(rna, "layers")
counts <- layers[["counts"]]

cat("Counts class:", class(counts), "\n")
cat("Counts dims:", dim(counts), "\n")

# Get gene names and cell barcodes from Dimnames
dimnames_list <- attr(counts, "Dimnames")
genes <- dimnames_list[[1]]
cells <- dimnames_list[[2]]

cat("Genes:", length(genes), "first:", head(genes, 5), "\n")
cat("Cells:", length(cells), "first:", head(cells, 3), "\n")

# If gene names are NULL, get from features
if (is.null(genes)) {
    features <- attr(rna, "features")
    # Try rownames
    cat("Features class:", class(features), "\n")
    cat("Features attr names:", names(attributes(features)), "\n")
}

if (is.null(cells)) {
    cells_attr <- attr(rna, "cells")
    cat("Cells attr class:", class(cells_attr), "\n")
}

# Get metadata
meta <- obj@meta.data
cat("\nMetadata dims:", dim(meta), "\n")
cat("Metadata columns:", colnames(meta), "\n")

# Save counts as MatrixMarket format (most portable)
cat("\nSaving counts matrix (MatrixMarket format)...\n")
# First assign proper dimnames to the matrix
if (!is.null(genes) && !is.null(cells)) {
    rownames(counts) <- genes
    colnames(counts) <- cells
}
writeMM(counts, "/home/jw3514/Work/Geneformer/data/QiuyanLiver/counts.mtx")

# Save gene names
cat("Saving gene names...\n")
write.csv(data.frame(gene = genes), "/home/jw3514/Work/Geneformer/data/QiuyanLiver/genes.csv", row.names = FALSE)

# Save cell barcodes
cat("Saving cell barcodes...\n")
write.csv(data.frame(barcode = cells), "/home/jw3514/Work/Geneformer/data/QiuyanLiver/barcodes.csv", row.names = FALSE)

# Save metadata
cat("Saving metadata...\n")
write.csv(meta, "/home/jw3514/Work/Geneformer/data/QiuyanLiver/metadata.csv")

cat("\nDone! Files saved.\n")
cat("Counts nnz:", length(counts@x), "\n")
cat("Counts value range:", range(counts@x), "\n")
