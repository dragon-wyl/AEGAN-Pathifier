rm(list = ls())
pcr <-  read.csv(file = "gan_GSE25066_pCR.csv")
rd <- read.csv(file = "GSE25066_RD.csv")
GSE25066 <- cbind(pcr[1:nrow(pcr), 2:ncol(pcr)], rd[1:nrow(rd), 2:ncol(rd)])
samples <- colnames(GSE25066)
allgenes = pcr$X
x <- NULL
x$title = "GSE25066数据"
new_row_pcr <- rep(FALSE, ncol(pcr) - 1)
new_row_rd <- rep(TRUE, ncol(rd) - 1)
normals <- c(new_row_pcr, new_row_rd)
x$data <- matrix(do.call(cbind, GSE25066[1:nrow(GSE25066), 1:ncol(GSE25066)]), nrow=nrow(GSE25066))
x$normals <- normals
x$allgenes <- allgenes
x$samples <- samples
f = 'GSE25066_exprs.Rdata'
save(x, file = f)

load(file = f)
