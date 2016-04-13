library(xgboost)
library(data.table)

features.tr.xgb <- read.csv('features_tr_xgb.csv')
features.ts.xgb <- read.csv('features_ts_xgb.csv')  
labels.train <- read.csv('label_train_sessions.csv')
labels.train <- as.factor(labels.train[,1])
labels.tr.xgb <- unclass(labels.train) - 1
classes <- levels(labels.train) 
id.test <- read.csv('id_test_sessions.csv')
id.test <- as.vector(id.test[,1])
id.train <- read.csv('id_train_sessions.csv')
id.train <- as.vector(id.train[,1])

par.xgb <- list('objective' = 'multi:softprob',
                'eval_metric' = 'ndcg5',
                'num_class' = length(classes))

cv.nround <- 50
cv.nfold <- 5

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = as.matrix(features.tr.xgb), 
                       label = labels.tr.xgb, 
                       eta = 0.1,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = cv.nround, 
                       subsample = 0.7,
                       colsample_bytree = 0.8,
                       seed = 1,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model <- xgboost(data = as.matrix(features.tr.xgb), 
                     label = labels.tr.xgb, 
                     eta = 0.1,
                     max_depth = 8, 
                     nround = 20, 
                     subsample = 0.7,
                     colsample_bytree = 0.8,
                     seed = 1,
                     eval_metric = ndcg5,
                     objective = "multi:softprob",
                     num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.fit <- predict(xgb.model, as.matrix(features.ts.xgb))
Sys.time() - tic

# generate submission
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

write.csv(sub.xgb.raw, file = 'sub13.csv', row.names = FALSE)

# ================ xgb with features with high importance ===============
matrix.importance <- xgb.importance(colnames(features.tr.xgb), model = xgb.model)
xgb.plot.importance(matrix.importance)

getFeaturesIndex <- function(features.original, features.new) {
  indice <- c()
  for (feature in features.original) {
    if (feature %in% features.new) {
      indice <- c(indice, which(features.original == feature))
    }
  }
  return(indice)
}

for (num in 5:15) {
  indice.temp <- getFeaturesIndex(colnames(features.tr.xgb), 
                                  matrix.importance$Feature[1:(num*10)])
  
  tic <- Sys.time()
  xgb.model.cv <- xgb.cv(param = par.xgb, 
                         data = as.matrix(features.tr.xgb[,indice.temp]),
                         label = labels.tr.xgb,
                         nfold = cv.nfold,
                         nrounds = cv.nround)
  Sys.time() - tic
  print(num)
}

for (num in 5:15) {
  indice.temp <- getFeaturesIndex(colnames(features.tr.xgb), 
                                  matrix.importance$Feature[1:(num*10)])
  
  tic <- Sys.time()
  xgb.model.cv <- xgb.cv(param = par.xgb, 
                         data = as.matrix(features.tr.xgb[,indice.temp]),
                         label = labels.tr.xgb,
                         nfold = cv.nfold,
                         nrounds = cv.nround)
  Sys.time() - tic
  print(num)
}

# fit
indice.fea <- getFeaturesIndex(colnames(features.tr.xgb), 
                               matrix.importance$Feature[1:80])

tic <- Sys.time()
xgb.fine.model <- xgboost(param = par.xgb,
                          data = as.matrix(features.tr.xgb[,indice.fea]),
                          label = labels.tr.xgb,
                          nrounds = 50)
Sys.time() - tic

tic <- Sys.time()
xgb.fine.fit <- predict(xgb.fine.model, as.matrix(features.ts.xgb))
Sys.time() - tic

# generate submission
sub.xgb.fine <- read.csv('sub1.csv')
sub.xgb.fine[,2] <- as.vector(sub.xgb.fine[,2])

tic <- Sys.time()
for (i in 1:dim(test)[1]) {
  id.temp <- as.character(test$id[i])
  if (id.temp %in% id.test) {
    ind.temp <- which(id.test == id.temp)
    prob.temp <- xgb.fine.fit[(12*ind.temp-11):(12*ind.temp)]
    table.temp <- cbind(prob.temp, classes)
    cou.temp <- table.temp[order(table.temp[,1], decreasing = TRUE), 2][1:5]
    sub.xgb.fine[(i*5-4):(i*5), 2] <- as.vector(cou.temp)
  }
}
Sys.time() - tic

write.csv(sub.xgb.fine, file = 'sub9.csv', row.names = FALSE)

