library(pathifier)
library(parallel)
library(parallel)
rm(list = ls())
#cores <- detectCores()
#cl <- makeCluster(cores)
#clusterExport(cl, list("quantify_pathways_deregulation"))
setwd("~/R-workspace/geo-analysis/GSE20194")
load(file = "GSE20194_exprs.Rdata")
load(file = "my_pathways.Rdata")
# load(file = "PDS_GSE25066.Rdata")
# data(KEGG)
min_exp <- min(x$data)
row_sd_vals <- apply(x$data, 1, sd)
min_std <- min(row_sd_vals)

PDS <- quantify_pathways_deregulation(x$data, x$allgenes, my_pathways$gs,
                                      my_pathways$pathwaynames,
                                      x$normals,
                                      attempts = 100,
                                      min_exp = min_exp,
                                      min_std = min_std
                                      )
e<-NULL
e$RD<-PDS$scores$MISMATCH_REPAIR[x$normals]
e$pCR<-PDS$scores$MISMATCH_REPAIR[!x$normals]
boxplot(e,ylab="score",main = "Pathway of MISMATCH_REPARI")
y <- NULL
y$RD <- PDS$scores$REGULATION_OF_AUTOPHAGY[x$normals]
y$pCR <- PDS$scores$REGULATION_OF_AUTOPHAGY[!x$normals]
boxplot(y,ylab="score",main = "Pathway of REGULATION_OF_AUTOPHAGY")

save(PDS, file = "PDS_GSE20194.Rdata")
as.character(x$samples[PDS$scores$REGULATION_OF_AUTOPHAGY>0.8])
PDS$center
curve <- PDS$curves$MISMATCH_REPAIR
curve <- cbind(curve, x$normals)
colnames(curve) <- c("PC1", "PC2", "PC3", "PC4", "PC5", "Group")
write.csv(curve, file = "GSE25066_PATHWAY_MR.csv", row.names = FALSE)
