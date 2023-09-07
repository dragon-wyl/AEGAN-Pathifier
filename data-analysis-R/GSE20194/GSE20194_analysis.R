library(GEOquery)
library(tidyverse)
library(stringr)
library(stringi)
setwd("~/R-workspace/geo-analysis/GSE20194")

rm(list = ls())
f = 'GSE20194_eSet.Rdata'
if (!file.exists(f)) {
  dat = getGEO(GEO = 'GSE20194', destdir = '.', AnnotGPL = TRUE, getGPL = TRUE)
  save(dat, file = f)
}
load(file = f)
pd <-  pData(dat[[1]])
exprs <- exprs(dat[[1]])

# 分组
colnames(pd)
group <- 
  ifelse(
    grepl('RD', pd$characteristics_ch1.2), 'RD',
      ifelse(grepl('pCR', pd$characteristics_ch1.2), 'pCR', 
        ifelse(grepl('RD', pd$characteristics_ch1.3), 'RD', 
          ifelse(grepl('pCR', pd$characteristics_ch1.3), 'pCR',
            ifelse(
              grepl('pCR', pd$characteristics_ch1.4), 'pCR',
              ifelse(
                grepl('RD', pd$characteristics_ch1.4), 'RD',
                'other'
              )
            )
                 
          )
          
        )  
      )
)
# 读取GPL平台信息文件
annot <- read.delim("./GPL96.annot",stringsAsFactors=FALSE,skip = 27) 
table(group)
# 选取行名并提取探针的基因名
exprs_rownames <-  rownames(exprs)
# 定义获取探针的基因名函数
get_tanzhen_genes <- function(tanzhen, ant) {
  exprs_gens <- NULL
  print(length(tanzhen))
  for (i in 1:length(tanzhen)) {

    # 使用apply函数遍历行
    row <- ant[i, ]  # 获取第i行
    x1 <- paste(tanzhen[i], row$ID, row$Gene.symbol, sep = ":")
    print(x1)
    exprs_gens <- append(exprs_gens, row$Gene.symbol)
    
  }
  return(exprs_gens)
  
}

exprs_genes <- get_tanzhen_genes(exprs_rownames, annot)
rownames(exprs) <- exprs_genes
exprs <- rbind(exprs, group)
deleted_rownames <- function(exprs_genes) {
  r <- NULL
  r <- append(r, '')
  for (i in 1:length(exprs_genes)) {
    if (grepl('///', exprs_genes[i]) || grepl('\\.', exprs_genes[i])){
      print(exprs_genes[i])
      r <- append(r, exprs_genes[i])
    } 
  }
  return(r)
}
# 将行名设置为数据框的行索引
# rows_to_delete <- deleted_rownames(exprs_genes)
dim(exprs)
exprs <- as.data.frame(exprs)
#row_indices <- which(rownames(exprs) %in% rows_to_delete)

# 删除指定行索引的行
#df <- df[-row_indices, ]
pCR <- 'pCR'
RD <- 'RD'
# 按行group选取列
selected_pCR <- exprs[, exprs['group', ] == pCR]
selected_RD <- exprs[, exprs['group', ] == RD]


# 将结果保存为csv文件
write.csv(selected_pCR, "GSE20194_pCR.csv", row.names=TRUE)
write.csv(selected_RD, "GSE20194_RD.csv", row.names=TRUE)

