rm(list = ls())
load(file = "GSE25066_exprs.Rdata")
load(file = "my_pathways.Rdata")
load(file = "PDS_GSE25066.Rdata")
GSE25066 <- NULL
for (score in PDS$scores) {
  GSE25066 <-  rbind(GSE25066, score)
}
GSE25066 <- rbind(GSE25066, x$normals)
GSE25066 <- t(GSE25066)
col_names <- sprintf("P%d",1:295)
col_names <- append(col_names, "group")
colnames(GSE25066) <- col_names
write.csv(GSE25066, "GSE25066_Pathways.csv",row.names = FALSE)
PDS$scores
pdf("my_boxplots.pdf")  # 创建一个新的 PDF 文件

for (i in seq_along(PDS$scores)) {
  score <- PDS$scores[[i]]
  e <- NULL
  e$RD <- score[x$normals]
  e$pCR <- score[!x$normals]
  boxplot(e, ylab = "score", main = i)
}

dev.off()  # 关闭 PDF 设备并保存文件

