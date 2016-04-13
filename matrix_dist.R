d1 <- readRDS('distances.rds')
d2 <- readRDS('distances.desk.rds')
d3 <- readRDS('distances.1.rds')
d4 <- readRDS('distances.desk.1.rds')
d5 <- readRDS('distances.desk.2.rds')
d6 <- readRDS('distances.2.rds')
d7 <- readRDS('distances.desk.3.rds')
d8 <- readRDS('distances.desk.4.rds')

l <- c(1, 15000, 15001, 30000, 30001, 50000, 50001, 60000,
       60001, 70000, 70001, 80000, 80001, 100000, 100001, 107911)

for (i in 1:8) {
  ind <- l[2*i]
  bottom <- dist2me(features.dist.NA[ind, ], features.dist.no.NA, 0.2)[1:10]
  print(ind)
  print(bottom)
}

d1[[15000]][1:10]
d2[[15000]][1:10]
d3[[20000]][1:10]
d4[[10000]][1:10]
d5[[10000]][1:10]
d6[[10000]][1:10]
d7[[20000]][1:10]
d8[[7911]][1:10]

getNeighbors <- function(aList, num) {
  m <- matrix(unlist(aList), 27572)
  m <- apply(m, 2, function(x) order(x)[1:num])
  return(t(m))
}

d1 <- readRDS('distances.rds')
system.time(m1 <- getNeighbors(d1, 1000))
d1 <- NULL

d2 <- readRDS('distances.desk.rds')
m2 <- getNeighbors(d2, 1000)
d2 <- NULL

d3 <- readRDS('distances.1.rds')
m3 <- getNeighbors(d3, 1000)
d3 <- NULL

d4 <- readRDS('distances.desk.1.rds')
m4 <- getNeighbors(d4, 1000)
d4 <- NULL

d5 <- readRDS('distances.desk.2.rds')
m5 <- getNeighbors(d5, 1000)
d5 <- NULL

d6 <- readRDS('distances.2.rds')
m6 <- getNeighbors(d6, 1000)
d6 <- NULL

d7 <- readRDS('distances.desk.3.rds')
m7 <- getNeighbors(d7, 1000)
d7 <- NULL

d8 <- readRDS('distances.desk.4.rds')
m8 <- getNeighbors(d8, 1000)
d8 <- NULL

matrix.dist <- rbind(m1, m2, m3, m4, m5, m6, m7, m8)
