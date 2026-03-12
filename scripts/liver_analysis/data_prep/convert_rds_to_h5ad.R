library(Seurat)
library(SeuratDisk)

cat("Reading RDS file...\n")
obj <- readRDS("/home/jw3514/Work/Geneformer/data/QiuyanLiver/04_humnucseq_scrublet_dim30_celltype_doubletremoved_integrated.rds")

cat("Object class:", class(obj), "\n")
cat("Dimensions:", dim(obj), "\n")
cat("Assays:", Assays(obj), "\n")
cat("Metadata columns:", colnames(obj@meta.data), "\n")
cat("First few rows of metadata:\n")
print(head(obj@meta.data, 3))

# Check what assay has raw counts
default_assay <- DefaultAssay(obj)
cat("\nDefault assay:", default_assay, "\n")

# Check available slots
for (assay_name in Assays(obj)) {
    cat("\nAssay:", assay_name, "\n")
    assay_obj <- obj[[assay_name]]
    cat("  Slots with data: ")
    if (!is.null(GetAssayData(obj, slot = "counts", assay = assay_name))) {
        cat("counts ")
        cat("(dims:", dim(GetAssayData(obj, slot = "counts", assay = assay_name)), ") ")
    }
    if (!is.null(GetAssayData(obj, slot = "data", assay = assay_name))) {
        cat("data ")
    }
    cat("\n")
}

# Save as h5Seurat then convert to h5ad
cat("\nSaving as h5Seurat...\n")
SaveH5Seurat(obj, filename = "/home/jw3514/Work/Geneformer/data/QiuyanLiver/liver.h5Seurat", overwrite = TRUE)

cat("Converting to h5ad...\n")
Convert("/home/jw3514/Work/Geneformer/data/QiuyanLiver/liver.h5Seurat", dest = "h5ad", overwrite = TRUE)

cat("Done!\n")
