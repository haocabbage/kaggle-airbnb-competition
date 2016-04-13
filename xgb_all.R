set.seed(1024)

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(X[,-1]), 
                       label = as.vector(y), 
                       eta = 0.1,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = cv.nround, 
                       subsample = 0.7,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = as.matrix(features.tr.xgb), 
                       label = labels.tr.xgb, 
                       eta = 0.1,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = cv.nround, 
                       subsample = 0.7,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(features.tr.all), 
                       label = labels.tr.xgb, 
                       eta = 0.05,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = 500, 
                       subsample = 0.9,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(features.tr.eho), 
                       label = labels.tr.xgb, 
                       eta = 0.05,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = 500, 
                       subsample = 0.9,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(fea.tr), 
                       label = labels.tr.xgb, 
                       eta = 0.1,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = cv.nround, 
                       subsample = 0.7,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(fea.tr.eho), 
                       label = labels.tr.xgb, 
                       eta = 0.1,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = cv.nround, 
                       subsample = 0.7,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(de.tr.eho), 
                       label = labels.tr.xgb, 
                       eta = 0.05,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = 500, 
                       subsample = 0.9,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

# learn
tic <- Sys.time()
xgb.model <- xgboost(data = data.matrix(features.tr.eho), 
                     label = labels.tr.xgb, 
                     eta = 0.02,
                     max_depth = 6, 
                     nround = 459, 
                     subsample = 0.8,
                     colsample_bytree = 0.8,
                     eval_metric = ndcg5,
                     objective = "multi:softprob",
                     num_class = 12)
Sys.time() - tic

# predict
tic <- Sys.time()
xgb.fit <- predict(xgb.model, data.matrix(features.ts.eho))
Sys.time() - tic

# generate submission
sub <- read.csv('sub1.csv')
sub[,2] <- as.vector(sub[,2])

tic <- Sys.time()
for (i in 1:dim(test)[1]) {
  id.temp <- as.character(test$id[i])
  if (id.temp %in% id.test) {
    ind.temp <- which(id.test == id.temp)
    prob.temp <- xgb.fit[(12*ind.temp-11):(12*ind.temp)]
    table.temp <- cbind(prob.temp, classes)
    cou.temp <- table.temp[order(table.temp[,1], decreasing = TRUE), 2][1:5]
    sub[(i*5-4):(i*5), 2] <- as.vector(cou.temp)
  }
}
Sys.time() - tic

# replace the last 3 predictions
sub.temp <- sub

tic <- Sys.time()
for (i in 1:dim(test)[1]) {
  id.temp <- as.character(test$id[i])
  if (id.temp %in% id.test) {
    ind.temp <- which(id.test == id.temp)
    cou.temp <- sub.temp[(i*5-4):(i*5), 2][1:2]
    cou1.temp <- top5[top5!=cou.temp[1] & top5!=cou.temp[2]]
    sub.temp[(i*5-4):(i*5), 2] <- c(cou.temp, cou1.temp[1:3])
  }
}
Sys.time() - tic

write.csv(sub, file = 'sub22.csv', row.names = FALSE)
write.csv(sub.temp, file = 'sub23.csv', row.names = FALSE)

# ================== combine one hot encoding features with session features ===============
indice.temp <- c()
for (i in 1:length(id.train)) {
  id.temp <- id.train[i]
  ind.temp <- which(X$id == id.temp)
  indice.temp <- c(indice.temp, ind.temp)
}

features.tr.all <- cbind(X[indice.temp,], features.tr.xgb[,11:167])[,-1]

indice1.temp <- c()
for (i in 1:length(id.test)) {
  id.temp <- id.test[i]
  ind.temp <- which(X_test$id == id.temp)
  indice1.temp <- c(indice1.temp, ind.temp)
}

features.ts.all <- cbind(X_test[indice1.temp,], features.ts.xgb[,11:167])[,-1]

xgb <- xgb.cv(data = data.matrix(X[,-1]), 
              label = as.vector(y), 
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

# =============== convert action time feature into ohe feature ==============
which(colnames(features.tr.all) == 'num.actions') #145

m.temp <- ifelse(features.tr.all[,146:301] != 0, 1, 0)
features.tr.eho <- cbind(features.tr.all[,1:145], m.temp)

m.temp <- ifelse(features.ts.all[,146:301] != 0, 1, 0)
features.ts.eho <- cbind(features.ts.all[,1:145], m.temp)

# ==================== remove constant features =======================
indice2.temp <- c()
for (i in 146:301) {
  f.temp <- features.tr.all[,i]
  f1.temp <- features.ts.all[,i]
  len.temp <- length(f.temp[f.temp != 0])
  len1.temp <- length(f1.temp[f1.temp != 0]) 
  if (len.temp == 0 | len1.temp == 0) indice2.temp <- c(indice2.temp, i)
}

fea.tr <- features.tr.all[,-indice2.temp]
fea.ts <- features.ts.all[,-indice2.temp]

fea.tr.eho <- features.tr.eho[,-indice2.temp]
fea.ts.eho <- features.ts.eho[,-indice2.temp]
