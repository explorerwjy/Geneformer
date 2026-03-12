library(Seurat)

cat("Reading RDS file...\n")
obj <- readRDS("/home/jw3514/Work/Geneformer/data/QiuyanLiver/04_humnucseq_scrublet_dim30_celltype_doubletremoved_integrated.rds")

cat("=== Object summary ===\n")
print(obj)

cat("\n=== All assays ===\n")
print(Assays(obj))

cat("\n=== Integrated assay details ===\n")
assay <- obj[["integrated"]]
cat("Class:", class(assay), "\n")
cat("Features:", nrow(assay), "\n")
cat("Cells:", ncol(assay), "\n")

# Check counts slot
counts_data <- GetAssayData(obj, slot = "counts", assay = "integrated")
cat("\nCounts slot class:", class(counts_data), "\n")
cat("Counts slot dims:", dim(counts_data), "\n")
cat("Counts slot sum:", sum(counts_data), "\n")

# Check data slot
data_slot <- GetAssayData(obj, slot = "data", assay = "integrated")
cat("\nData slot class:", class(data_slot), "\n")
cat("Data slot dims:", dim(data_slot), "\n")
cat("Data slot range:", range(data_slot), "\n")
cat("Data slot first 5 genes:", rownames(data_slot)[1:5], "\n")
cat("Data slot sample values:\n")
print(data_slot[1:5, 1:3])

# Check scale.data slot
scale_data <- GetAssayData(obj, slot = "scale.data", assay = "integrated")
cat("\nScale.data slot class:", class(scale_data), "\n")
cat("Scale.data slot dims:", dim(scale_data), "\n")

# Check metadata
cat("\n=== Metadata ===\n")
cat("Columns:", colnames(obj@meta.data), "\n")
cat("Cell types:\n")
print(table(obj@meta.data$Cell.type.new))
cat("\nSample distribution:\n")
print(table(obj@meta.data$orig.ident))

# Check if there's an RNA assay hiding somewhere
cat("\n=== Checking for RNA assay ===\n")
cat("Assay names:", Assays(obj), "\n")

# Check misc/commands for clues about original data
cat("\nSlot names in object:\n")
print(slotNames(obj))
