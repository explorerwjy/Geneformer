library(Seurat)
library(Matrix)

cat("Reading RDS file...\n")
obj <- readRDS("/home/jw3514/Work/Geneformer/data/QiuyanLiver/04_humnucseq_scrublet_dim30_celltype_doubletremoved_integrated.rds")

rna <- obj@assays[["RNA"]]

# Access via attributes since it's S4
cat("Attribute names:", names(attributes(rna)), "\n")

layers <- attr(rna, "layers")
cat("\nLayers class:", class(layers), "\n")
cat("Layer names:", names(layers), "\n")

for (lname in names(layers)) {
    layer <- layers[[lname]]
    cat("\n--- Layer:", lname, "---\n")
    cat("Class:", class(layer), "\n")
    cat("Attr names:", names(attributes(layer)), "\n")

    # Try to get the matrix from inside
    layer_matrix <- attr(layer, "matrix")
    if (!is.null(layer_matrix)) {
        cat("Matrix class:", class(layer_matrix), "\n")
        cat("Matrix dims:", dim(layer_matrix), "\n")
    }

    # Just try to see all attributes
    for (aname in names(attributes(layer))) {
        a <- attr(layer, aname)
        cat("  ", aname, "-> class:", class(a))
        if (is.matrix(a) || inherits(a, "dgCMatrix")) {
            cat(" dims:", dim(a))
            if (inherits(a, "dgCMatrix")) {
                cat(" nnz:", length(a@x))
            }
        } else if (is.character(a) || is.numeric(a)) {
            cat(" len:", length(a), " head:", head(a, 3))
        }
        cat("\n")
    }
}

# Also check cells and features
cells <- attr(rna, "cells")
features <- attr(rna, "features")
cat("\n=== Cells ===\n")
cat("Class:", class(cells), "\n")
cat("Length/dim:", length(cells), dim(cells), "\n")
cat("Head:", head(cells, 3), "\n")

cat("\n=== Features ===\n")
cat("Class:", class(features), "\n")
if (is.data.frame(features)) {
    cat("Dims:", dim(features), "\n")
    cat("Colnames:", colnames(features), "\n")
    cat("Head rownames:", head(rownames(features)), "\n")
    print(head(features))
} else {
    cat("Length:", length(features), "\n")
    cat("Head:", head(features, 5), "\n")
}
