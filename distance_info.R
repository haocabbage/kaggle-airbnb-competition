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

SIZE <- NROW(features.dist)
task <- allocateTask(SIZE, 3762)
distances <- vector('list', 107911)

library(doSNOW)
cluster <- makeCluster(4, 'SOCK')

saveRDS(distances, file = 'distances.rds')


block <- SIZE%/%4
b1 <- 1:block
b2 <- (block+1):(2*block)
b3 <- (2*block+1):(3*block)
b4 <- (3*block+1):SIZE
blocks <- list(b1, b2, b3, b4)

tic <- Sys.time()
results <- foreach(i = 80001:85000) %dopar% dist2me(features.dist[i,], 
                                            features.dist[i:SIZE,], 
                                            0.2)
Sys.time() - tic

for (i in 1:5000) {
  distances[[i+80000]] <- results[[i]]
}

# xgb with features w/ NA
which(features.complete$id == as.character(features.sessions$id[1]))
ind.sessions <- c()

for (i in 1:NROW(features.sessions)) {
  id.temp <- as.character(features.sessions$id[i])
  ind.temp <- which(features.complete$id[137000:NROW(features.complete)] == id.temp)
  ind.sessions <- c(ind.sessions, ind.temp)
  print(i)
}

ind.sessions <- ind.sessions + 136999
saveRDS(ind.sessions, 'ind.sessions.rds')
features.with.NA <- cbind(features.complete[ind.sessions, 2:145], features.to.fill)

which(features.sessions$id == as.character(test$id[1]))
features.with.NA.tr <- features.with.NA[1:73815, ]
features.with.NA.ts <- features.with.NA[73816:NROW(features.with.NA), ]
id.test <- as.vector(features.sessions$id[73816:NROW(features.with.NA)])
labels <- train$country_destination[ind.sessions[1:73815]]
labels.num <- unclass(labels)-1

tic <- Sys.time()
xgb.model <- xgb.cv(data = data.matrix(features.with.NA.tr), 
                    label = labels.num, 
                    eta = 0.05,
                    max_depth = 8,
                    nfold = 5,
                    nrounds = 500, 
                    subsample = 0.9,
                    colsample_bytree = 0.8,
                    eval_metric = ndcg5,
                    objective = "multi:softprob",
                    num_class = 12,
                    missing = NA)
Sys.time() - tic

tic <- Sys.time()
xgb.model <- xgboost(data = data.matrix(features.with.NA.tr), 
                     label = labels.num, 
                     eta = 0.05,
                     max_depth = 8,
                     nrounds = 85, 
                     subsample = 0.9,
                     colsample_bytree = 0.8,
                     eval_metric = ndcg5,
                     objective = "multi:softprob",
                     num_class = 12,
                     missing = NA)
Sys.time() - tic

tic <- Sys.time()
xgb.fit <- predict(xgb.model, data.matrix(features.with.NA.ts), missing = NA)
Sys.time() - tic

# generate submission
classes <- as.vector(levels(train$country_destination))
sub.xgb.raw <- read.csv('sub1.csv')
sub.xgb.raw[,2] <- as.vector(sub.xgb.raw[,2])

tic <- Sys.time()
for (i in 1:dim(test)[1]) {
  id.temp <- as.character(test$id[i])
  if (id.temp %in% id.test) {
    ind.temp <- which(id.test == id.temp)
    prob.temp <- xgb.fit[(12*ind.temp-11):(12*ind.temp)]
    table.temp <- cbind(prob.temp, classes)
    cou.temp <- table.temp[order(table.temp[,1], decreasing = TRUE), 2][1:5]
    sub.xgb.raw[(i*5-4):(i*5), 2] <- as.vector(cou.temp)
  }
}
Sys.time() - tic

write.csv(sub.xgb.raw, file = 'sub25.csv', row.names = FALSE)
