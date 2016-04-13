tr.eho <- de.tr.eho[,-(147:302)]
ts.eho <- de.ts.eho[,-(147:302)]
saveRDS(rbind(tr.eho, ts.eho), file = 'features.xgb.rds')

# ==========================================================
library(doParallel)
cl <- makeCluster(3)
registerDoParallel(cl)

xgb.models <- foreach(i=1:50, .packages='xgboost') %dopar% {
  xgboost_bagging(tr.eho, labels.num)  
}

stopCluster(cl)

# ===========================================================
library(xgboost)
tr.eho <- features.xgb[1:73815, ]
ts.eho <- features.xgb[73816:135483, ]
set.seed(42)

xgb.models <- list()

for (i in 31:50) {
  xgb.models[[i]] <- xgboost_bagging(tr.eho, labels.num)
}

saveRDS(xgb.models, file = 'xgb.models.rds')

xgb.fit <- predict_xgb_bagging(xgb.models, ts.eho)

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

write.csv(sub.xgb.raw, file = 'sub44.csv', row.names = FALSE)

# ==================================================================
data <- read.csv('data.csv')
data.log <- read.csv('data_log.csv')
data.tr <- data[1:73815, ]
data.log.tr <- data.log[1:73815, ]

for (i in 1:40) {
  data.models[[i]] <- data_bagging(data.tr, labels.num, 30)
}

for (i in 1:40) {
  data.log.models[[i]] <- data_bagging(data.log.tr, labels.num, 25)
}

pred.xgb <- predict_xgb_bagging(xgb.models.40, features.xgb)
pred.data <- predict_data_bagging(data.models, data)
pred.data.log <- predict_data_bagging(data.log.models, data.log)

m.xgb <- matrix(pred.xgb, 12)
m.data <- matrix(pred.data, 12)
m.data.log <- matrix(pred.data.log, 12)

m.xgb <- apply(m.xgb, 2, function(x) x/sum(x))
m.data <- apply(m.data, 2, function(x) x/sum(x))
m.data.log <- apply(m.data.log, 2, function(x) x/sum(x))
m.xgb <- t(m.xgb)
m.data <- t(m.data)
m.data.log <- t(m.data.log)

saveRDS(m.xgb, file = 'm.xgb.rds')
saveRDS(m.data, file = 'm.data.rds')
saveRDS(m.data.log, file = 'm.data.log.rds')

# ==================================================================
xgboost_bagging <- function(data, labels) {
  # bootstrap sampling
  ind.sp <- sample(1:73815, 73815, replace = TRUE)
  sp.boot <- data[ind.sp,]
  labels.boot <- labels[ind.sp]
  
  # build classifier
  model.temp <- xgboost(data = data.matrix(sp.boot[,-1]), 
                        label = labels.boot, 
                        eta = 0.05,
                        max_depth = 8,
                        nrounds = 150, 
                        subsample = 0.9,
                        colsample_bytree = 0.8,
                        eval_metric = ndcg5,
                        objective = "multi:softprob",
                        num_class = 12)
  return(model.temp)
}

predict_xgb_bagging <- function (models, data) {
  # bagging prediction
  xgb.preds <- c()
  for (i in 1:length(models)) {
    pred <- predict(models[[i]], data.matrix(data[,-1]))
    xgb.preds <- cbind(xgb.preds, pred)
  }
  xgb.fit <- apply(xgb.preds, 1, function(x) 1/mean(1/x))
  return(xgb.fit)
}

data_bagging <- function(data, labels, rounds) {
  # bootstrap sampling
  ind.sp <- sample(1:NROW(data), NROW(data), replace = TRUE)
  sp.boot <- data[ind.sp,]
  labels.boot <- labels[ind.sp]
  
  # build classifier
  model.temp <- xgboost(data = data.matrix(sp.boot), 
                        label = labels.boot, 
                        eta = 0.05,
                        max_depth = 8,
                        nrounds = rounds, 
                        subsample = 0.9,
                        colsample_bytree = 0.8,
                        eval_metric = ndcg5,
                        objective = "multi:softprob",
                        num_class = 12)
  return(model.temp)
}

predict_data_bagging <- function (models, data) {
  # bagging prediction
  xgb.preds <- c()
  for (i in 1:length(models)) {
    pred <- predict(models[[i]], data.matrix(data))
    xgb.preds <- cbind(xgb.preds, pred)
  }
  xgb.fit <- apply(xgb.preds, 1, function(x) 1/mean(1/x))
  return(xgb.fit)
}