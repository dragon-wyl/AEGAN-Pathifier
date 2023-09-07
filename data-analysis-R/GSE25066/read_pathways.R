file_names <- sprintf("%d.txt",1:298)
gs <- list()
pathway_names <- list()
for (file_name in file_names) {
  file_path <- file.path("./new_pathway", file_name)  # 构建文件路径
  gs[[file_name]] <- readLines(file_path)  # 读取文件内容
  pathway_names[[file_name]] <- file_name
}

my_pathways <- list(gs = gs, pathwaynames = pathway_names)
save(my_pathways, file = "my_pathways.Rdata")
