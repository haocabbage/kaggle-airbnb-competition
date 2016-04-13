library(foreach)
library(iterators)
library(snow)
library(doSNOW)

cl <- makeCluster(4)
registerDoSNOW(cl)

par.svm <- list()
par.svm[[1]] <- c(0.1, 1e-10)
par.svm[[2]] <- c(1000, 1e-10)
par.svm[[3]] <- c(100, 1e-12)
par.svm[[4]] <- c(100, 1e-11)

par.rf <- list()
par.rf[[1]] <- c(10, 2000)
par.rf[[2]] <- c(20, 2000)
par.rf[[3]] <- c(50, 2000)
par.rf[[4]] <- c(20, 500)

tic <- Sys.time()
svm.models <- foreach(i=1:length(par.svm), .packages = 
                        'e1071') %dopar% {svm(train.features[,-1], labels.bi,
                                              cost=par.svm[[i]][1],
                                              gamma=par.svm[[i]][2])}
Sys.time() - tic

tic <- Sys.time()
svm.fits <- foreach(i=1:length(par.svm), .packages = 
                        'e1071') %dopar% {predict(svm.models[[i]],
                                                  test.features[,-1])}
Sys.time() - tic

tic <- Sys.time()
rf.models <- foreach(i=1:length(par.rf), .packages = 
                        'randomForest') %dopar% {randomForest(train.features[,-1], 
                                                              labels.bi, 
                                                              importance=TRUE, 
                                                              mtry=par.rf[[i]][1],
                                                              ntree=par.rf[[i]][2])}
Sys.time() - tic

tic <- Sys.time()
rf.fits <- foreach(i=1:length(par.rf), .packages = 
                        'randomForest') %dopar% {predict.randomForest(rf.models[[i]],
                                                     test.features[,-1])}
Sys.time() - tic

stopCluster(cl)

svm.models <- list(1,2,3,4)
rf.models <-list(1,2,3,4)
svm.fits <- list(1,2,3,4)
rf.fits <- list(1,2,3,4)

for (i in 3:4) {
  tic <- Sys.time()
  svm.models[[i]] <- svm(train.features[,-1], labels.bi,
                         cost=par.svm[[i]][1],
                         gamma=par.svm[[i]][2])
  print(Sys.time() - tic)
  
  tic <- Sys.time()
  svm.fits[[i]] <- predict(svm.models[[i]],
                           test.features[,-1])
  print(Sys.time() - tic)
  
  print(table(svm.fits[[i]]))
}

for (i in 1:4) {
  tic <- Sys.time()
  rf.models[[i]] <- randomForest(train.features[,-1], 
                                 labels.bi, 
                                 importance=TRUE, 
                                 mtry=par.rf[[i]][1],
                                 ntree=par.rf[[i]][2])
  print(Sys.time() - tic)
  
  tic <- Sys.time()
  rf.fits[[i]] <- predict.randomForest(rf.models[[i]],
                           test.features[,-1])
  print(Sys.time() - tic)
  
  print(table(rf.fits[[i]]))
}


# ==================== mlp =====================
library(h2o)
localH2O = h2o.init(nthreads = -1, max_mem_size = '20g')

training <- as.data.frame(cbind(labels.bi, train.features[,-1]))
training_h2o <- as.h2o(training, "training")
test_h2o <- as.h2o(as.data.frame(test.features[,-1]), "test")

mlp <- function (node, epoch, key, test) {
  tic <- Sys.time()
  model <- 
    h2o.deeplearning(x = 2:158,  # column numbers for predictors
                     y = 1,   # column number for label
                     training_frame = key, # data in H2O format
                     activation = "RectifierWithDropout", # or 'Tanh'
                     input_dropout_ratio = 0.2, # % of inputs dropout
                     hidden_dropout_ratios = c(0.5), # % for nodes dropout
                     balance_classes = FALSE, 
                     hidden = c(node), # one layer
                     epochs = epoch) # max. no. of epochs
  print(Sys.time() - tic)
  
  temp.pred <- h2o.predict(model, test)
  temp.pred <- as.factor(as.data.frame(temp.pred)[,1])
  
  return(temp.pred)
}

temp <- mlp(600, 1000, 'training', test_h2o)

table(temp)

mlp.fits <- list()
for (i in 1:5) {
  for (j in 1:10) {
    temp <- mlp(25*i, 100*i, 'training', test_h2o)
    mlp.fits[[i*j]] <- temp
    print(table(temp))
  }
}

