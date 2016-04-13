getDist <- function(a, b, lambda) {
  # compute distance between two data points a and b
  temp <- abs(a - b) - lambda
  dis <- sum(temp, na.rm = TRUE) + length(a)*lambda
  return(dis)
}

dist2me <- function (a, B, LAMBDA) {
  # compute distance between point a and points in B
  if (is.vector(B)) return(getDist(a, B, LAMBDA))
  distances <- apply(B, 1, function(x) getDist(a, x, LAMBDA))
  return(distances)
}

allocateTask <- function(len, num) {
  # allocate points into designated number of clusters
  cluster <- vector('list', num)
  round <- len%/%(num*2)
  left <- len - round*num*2
  
  for (i in 1:round) {
    top <- (num*i - num + 1):(num*i)  
    bottom <- (len - num*i + num):(len - num*i + 1)
    for (j in 1:num) {
      cluster[[j]] <- c(cluster[[j]], top[j], bottom[j])
    }
  }
  
  for (k in 1:left) {
    if (k == num) {
      cluster[[k]] <- c(cluster[[k]], num*round + k)
    } else {
      cluster[[k%%num]] <- c(cluster[[k%%num]], num*round + k)  
    }
  }
  
  return(cluster)
}

features.dist <- readRDS('features.dist.rds')
SIZE <- NROW(features.dist)
ind.NA <- readRDS('ind.NA.rds')
features.dist.NA <- features.dist[1:length(ind.NA), ]
features.dist.no.NA <- features.dist[(length(ind.NA)+1):SIZE, ]

library(doParallel)
cluster <- makeCluster(3)
registerDoParallel(cluster)

system.time({
  distances.desk.3 <- foreach(i = 80001:100000) %dopar% {
    dist2me(features.dist.NA[i,], features.dist.no.NA, 0.2)
  }  
})

saveRDS(distances.desk.3, file = 'distances.desk.3.rds')
stopCluster(cluster)


