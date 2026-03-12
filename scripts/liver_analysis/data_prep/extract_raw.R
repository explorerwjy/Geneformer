library(Seurat)

cat("Reading RDS file...\n")
obj <- readRDS("/home/jw3514/Work/Geneformer/data/QiuyanLiver/04_humnucseq_scrublet_dim30_celltype_doubletremoved_integrated.rds")

rna <- obj@assays[["RNA"]]

# Try to access Assay5 internals using attributes/unclass
cat("=== Trying to access Assay5 internals ===\n")
cat("typeof:", typeof(rna), "\n")
cat("attributes names:", names(attributes(rna)), "\n")

# Unclass to see raw structure
raw_rna <- unclass(rna)
cat("After unclass, typeof:", typeof(raw_rna), "\n")
cat("Names:", names(raw_rna), "\n")

if (is.list(raw_rna)) {
    for (nm in names(raw_rna)) {
        el <- raw_rna[[nm]]
        cat("\n  ", nm, "-> class:", class(el))
        if (is.matrix(el) || inherits(el, "dgCMatrix") || inherits(el, "Matrix")) {
            cat(" dims:", dim(el))
        } else if (is.list(el)) {
            cat(" names:", names(el), " length:", length(el))
        } else if (is.vector(el)) {
            cat(" length:", length(el), " head:", head(el, 3))
        }
    }
    cat("\n")
}

# Try accessing layers if it exists
if ("layers" %in% names(raw_rna)) {
    cat("\n=== Layers ===\n")
    layers <- raw_rna[["layers"]]
    cat("Layer names:", names(layers), "\n")
    for (lname in names(layers)) {
        layer <- layers[[lname]]
        cat("\nLayer:", lname, "\n")
        cat("  class:", class(layer), "\n")
        # Try to unclass the layer too
        raw_layer <- unclass(layer)
        cat("  typeof:", typeof(raw_layer), "\n")
        if (is.list(raw_layer)) {
            cat("  names:", names(raw_layer), "\n")
            for (subnm in names(raw_layer)) {
                subel <- raw_layer[[subnm]]
                cat("    ", subnm, "-> class:", class(subel))
                if (is.matrix(subel) || inherits(subel, "dgCMatrix") || inherits(subel, "Matrix")) {
                    cat(" dims:", dim(subel))
                    if (inherits(subel, "dgCMatrix")) {
                        cat(" nnz:", length(subel@x), " range:", range(subel@x))
                        cat("\n    first genes:", head(rownames(subel)))
                    }
                }
                cat("\n")
            }
        }
    }
}
