library(Seurat)

cat("Reading RDS file...\n")
obj <- readRDS("/home/jw3514/Work/Geneformer/data/QiuyanLiver/04_humnucseq_scrublet_dim30_celltype_doubletremoved_integrated.rds")

cat("=== Full object structure ===\n")
cat("Class:", class(obj), "\n")
cat("Assays from names(obj@assays):", names(obj@assays), "\n")

# Check each assay thoroughly
for (aname in names(obj@assays)) {
    cat("\n--- Assay:", aname, "---\n")
    a <- obj@assays[[aname]]
    cat("Class:", class(a), "\n")
    cat("Slot names:", slotNames(a), "\n")

    # counts
    ct <- a@counts
    cat("counts: class=", class(ct), " dims=", dim(ct), "\n")
    if (length(ct) > 0 && prod(dim(ct)) > 0) {
        cat("  range:", range(ct@x), "\n")
        cat("  first genes:", head(rownames(ct)), "\n")
        cat("  nnz:", length(ct@x), "\n")
    }

    # data
    dt <- a@data
    cat("data: class=", class(dt), " dims=", dim(dt), "\n")
    if (length(dt) > 0 && prod(dim(dt)) > 0) {
        cat("  range:", range(dt@x), "\n")
        cat("  first genes:", head(rownames(dt)), "\n")
        cat("  nnz:", length(dt@x), "\n")
    }

    # scale.data
    sd <- a@scale.data
    cat("scale.data: class=", class(sd), " dims=", dim(sd), "\n")

    # key, var.features
    cat("key:", a@key, "\n")
    cat("n var.features:", length(a@var.features), "\n")
}

# Also check misc, tools, commands for stored data
cat("\n=== Misc ===\n")
cat("names(obj@misc):", names(obj@misc), "\n")
cat("names(obj@tools):", names(obj@tools), "\n")
cat("names(obj@commands):", names(obj@commands), "\n")
cat("names(obj@graphs):", names(obj@graphs), "\n")
cat("names(obj@neighbors):", names(obj@neighbors), "\n")

# Check if there's something in images or other slots
cat("\n=== str of top-level (depth=1) ===\n")
str(obj, max.level = 1)
