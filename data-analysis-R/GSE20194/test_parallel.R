rm(list=ls())
fun <- function(x){
  return (x+1);
}
system.time({
  res <- lapply(1:5000000, fun)
})
#加载parallel包
library(parallel)

#detectCores函数可以告诉你你的CPU可使用的核数
clnum<-detectCores() 

#设置参与并行的CPU核数目，这里我们使用了所有的CPU核，也就是我们刚才得到的clnum，具体到这个案例，clnum=4
cl <- makeCluster(getOption("cl.cores", clnum))
system.time({
  res <- parLapply(cl, 1:5000000, fun)
})
