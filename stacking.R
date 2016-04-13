setwd("~/Dropbox/airbnb")

library(xgboost)
library(h2o)
h2o.init(nthreads = -1, max_mem_size = '40G')
set.seed(42)

# data used: features.xgb, data
# data1: features.xgb
data1 <- readRDS('features.xgb.rds')[1:73815, -1]

# data2: data
data2 <- read.csv('data.csv')[1:73815, ]

# divide data used into 2 subsets, indices randomly picked
ind.a <- sort(sample(1:73815, 73815%/%2))
ind.b <- (1:73815)[-ind.a] 

saveRDS(ind.a, file = 'ind.a.rds')
saveRDS(ind.b, file = 'ind.b.rds')

ind.a <- as.vector(readRDS('ind.a.rds'))
ind.b <- as.vector(readRDS('ind.b.rds'))

data1.a <- data1[ind.a, ]
data1.b <- data1[ind.b, ]

data2.a <- data2[ind.a, ]
data2.b <- data2[ind.b, ]

labels <- readRDS('labels.rds')
labels.num <- unclass(labels) - 1

labels.a <- labels[ind.a]
labels.b <- labels[ind.b]

labels.num.a <- labels.num[ind.a]
labels.num.b <- labels.num[ind.b]

data1.a.h2o <- as.h2o(data1.a, destination_frame = 'data1.a.h2o')
data1.b.h2o <- as.h2o(data1.b, destination_frame = 'data1.b.h2o')
data2.a.h2o <- as.h2o(data2.a, destination_frame = 'data2.a.h2o')
data2.b.h2o <- as.h2o(data2.b, destination_frame = 'data2.b.h2o')

tr1.a.h2o <- as.h2o(cbind(labels.a, data1.a), destination_frame = 'tr1.a.h2o')
tr1.b.h2o <- as.h2o(cbind(labels.b, data1.b), destination_frame = 'tr1.b.h2o')
tr2.a.h2o <- as.h2o(cbind(labels.a, data2.a), destination_frame = 'tr2.a.h2o')
tr2.b.h2o <- as.h2o(cbind(labels.b, data2.b), destination_frame = 'tr2.b.h2o')

pred.temp <- matrix(0, 73815, 12)

# ================= xgb ===================

# data 1
#pred.data1.xgb <- pred.temp

# train a, predict b
xgb.models.1.a <- list()
for (i in 1:40) {
  xgb.models.1.a[[i]] <- data_bagging(data1.a, labels.num.a, 60)
}

xgb.temp <- predict_data_bagging(xgb.models.1.a, data1.b)
pred.data1.xgb[ind.b,] <- t(matrix(xgb.temp, 12))

# train b, predict a
xgb.models.1.b <- list()
for (i in 1:40) {
  xgb.models.1.b[[i]] <- data_bagging(data1.b, labels.num.b, 60)
}

xgb.temp <- predict_data_bagging(xgb.models.1.b, data1.a)
pred.data1.xgb[ind.a,] <- t(matrix(xgb.temp, 12))

# data 2
#pred.data2.xgb <- pred.temp

# train a, predict b
xgb.models.2.a <- list()
for (i in 1:40) {
  xgb.models.2.a[[i]] <- data_bagging(data2.a, labels.num.a, 25)
}

xgb.temp <- predict_data_bagging(xgb.models.2.a, data2.b)
pred.data2.xgb[ind.b,] <- t(matrix(xgb.temp, 12))

# train b, predict a
xgb.models.2.b <- list()
for (i in 1:40) {
  xgb.models.2.b[[i]] <- data_bagging(data2.b, labels.num.b, 25)
}

xgb.temp <- predict_data_bagging(xgb.models.2.b, data2.a)
pred.data2.xgb[ind.a,] <- t(matrix(xgb.temp, 12))

# ================= rf ===================

# data 1
pred.data1.rf <- pred.temp 

# train a, predict b
rf.model.1.a <- h2o.randomForest(training_frame = tr1.a.h2o, 
                                 x = 2:610,                       
                                 y = 1,                         
                                 mtries = 80, 
                                 ntrees = 200, 
                                 balance_classes = FALSE,
                                 max_depth = 40,               
                                 stopping_rounds = 5,          
                                 stopping_tolerance = 1e-2,
                                 seed = 42)

rf.temp <- h2o.predict(rf.model.1.a, data1.b.h2o)
rf.temp <- as.data.frame(rf.temp)[,c(2:10, 13, 11, 12)]
pred.data1.rf[ind.b,] <- t(apply(rf.temp, 1, function(x) (x + 1)/sum(x + 1)))

# train b, predict a
rf.model.1.b <- h2o.randomForest(training_frame = tr1.b.h2o, 
                                 x = 2:610,                       
                                 y = 1,                         
                                 mtries = 80, 
                                 ntrees = 200, 
                                 balance_classes = FALSE,
                                 max_depth = 40,               
                                 stopping_rounds = 5,          
                                 stopping_tolerance = 1e-2,
                                 seed = 42) 

rf.temp <- h2o.predict(rf.model.1.b, data1.a.h2o)
rf.temp <- as.data.frame(rf.temp)[,c(2:10, 13, 11, 12)]
pred.data1.rf[ind.a,] <- t(apply(rf.temp, 1, function(x) (x + 1)/sum(x + 1)))

# data 2
pred.data2.rf <- pred.temp

# train a, predict b
rf.model.2.a <- h2o.randomForest(training_frame = tr2.a.h2o, 
                                 x = 2:314,                       
                                 y = 1,                         
                                 mtries = 20, 
                                 ntrees = 200, 
                                 balance_classes = FALSE,
                                 max_depth = 20,               
                                 stopping_rounds = 5,          
                                 stopping_tolerance = 1e-2,
                                 seed = 42) 

rf.temp <- h2o.predict(rf.model.2.a, data2.b.h2o)
rf.temp <- as.data.frame(rf.temp)[,c(2:10, 13, 11, 12)]
pred.data2.rf[ind.b,] <- t(apply(rf.temp, 1, function(x) (x + 1)/sum(x + 1)))

# train b, predict a
rf.model.2.b <- h2o.randomForest(training_frame = tr2.b.h2o, 
                                 x = 2:314,                       
                                 y = 1,                         
                                 mtries = 20, 
                                 ntrees = 200, 
                                 balance_classes = FALSE,
                                 max_depth = 20,               
                                 stopping_rounds = 5,          
                                 stopping_tolerance = 1e-2,
                                 seed = 42) 

rf.temp <- h2o.predict(rf.model.2.b, data2.a.h2o)
rf.temp <- as.data.frame(rf.temp)[,c(2:10, 13, 11, 12)]
pred.data2.rf[ind.a,] <- t(apply(rf.temp, 1, function(x) (x + 1)/sum(x + 1)))

# ================= gbm ===================

# data 1
pred.data1.gbm <- pred.temp

# train a, predict b
gbm.model.1.a <- h2o.gbm(training_frame = tr1.a.h2o,     
                         x= 2:610,                     
                         y= 1,                        
                         ntrees = 50,                
                         learn_rate = 0.2,           
                         max_depth = 20,              
                         sample_rate = 0.7,          
                         col_sample_rate = 0.7,       
                         stopping_rounds = 2,         
                         stopping_tolerance = 0.01,  
                         score_each_iteration = T,   
                         seed = 42) 

gbm.temp <- h2o.predict(gbm.model.1.a, data1.b.h2o)
gbm.temp <- as.data.frame(gbm.temp)[,c(2:10, 13, 11, 12)]
pred.data1.gbm[ind.b,] <- t(apply(gbm.temp, 1, function(x) (x + 1)/sum(x + 1)))

# train b, predict a
gbm.model.1.b <- h2o.gbm(training_frame = tr1.b.h2o,     
                         x= 2:610,                     
                         y= 1,                        
                         ntrees = 50,                
                         learn_rate = 0.2,           
                         max_depth = 20,              
                         sample_rate = 0.7,          
                         col_sample_rate = 0.7,       
                         stopping_rounds = 2,         
                         stopping_tolerance = 0.01,  
                         score_each_iteration = T,   
                         seed = 42)  

gbm.temp <- h2o.predict(gbm.model.1.b, data1.a.h2o)
gbm.temp <- as.data.frame(gbm.temp)[,c(2:10, 13, 11, 12)]
pred.data1.gbm[ind.a,] <- t(apply(gbm.temp, 1, function(x) (x + 1)/sum(x + 1)))

# data 2
pred.data2.gbm <- pred.temp

# train a, predict b
gbm.model.2.a <- h2o.gbm(training_frame = tr2.a.h2o,     
                         x= 2:314,                     
                         y= 1,                        
                         ntrees = 50,                
                         learn_rate = 0.1,           
                         max_depth = 20,              
                         sample_rate = 0.7,          
                         col_sample_rate = 0.7,       
                         stopping_rounds = 2,         
                         stopping_tolerance = 0.01,  
                         score_each_iteration = T,   
                         seed = 42) 

gbm.temp <- h2o.predict(gbm.model.2.a, data2.b.h2o)
gbm.temp <- as.data.frame(gbm.temp)[,c(2:10, 13, 11, 12)]
pred.data2.gbm[ind.b,] <- t(apply(gbm.temp, 1, function(x) (x + 1)/sum(x + 1)))

# train b, predict a
gbm.model.2.b <- h2o.gbm(training_frame = tr2.b.h2o,     
                         x= 2:314,                     
                         y= 1,                        
                         ntrees = 50,                
                         learn_rate = 0.1,           
                         max_depth = 20,              
                         sample_rate = 0.7,          
                         col_sample_rate = 0.7,       
                         stopping_rounds = 2,         
                         stopping_tolerance = 0.01,  
                         score_each_iteration = T,   
                         seed = 42)  

gbm.temp <- h2o.predict(gbm.model.2.b, data2.a.h2o)
gbm.temp <- as.data.frame(gbm.temp)[,c(2:10, 13, 11, 12)]
pred.data2.gbm[ind.a,] <- t(apply(gbm.temp, 1, function(x) (x + 1)/sum(x + 1)))

h2o.shutdown()

# ======================================================

pred.data1.xgb <- rbind(pred.data1.xgb, m.xgb[73816:135483,])
pred.data2.xgb <- rbind(pred.data2.xgb, m.data[73816:135483,])
pred.data1.xgb <- t(apply(pred.data1.xgb, 1, function(x) x/sum(x)))
pred.data2.xgb <- t(apply(pred.data2.xgb, 1, function(x) x/sum(x)))

pred.data1.rf <- rbind(pred.data1.rf, pred.rf[73816:135483,])
pred.data2.rf <- rbind(pred.data2.rf, pred.data.rf[73816:135483,])

pred.data1.gbm <- rbind(pred.data1.gbm, pred.gbm[73816:135483,])
pred.data2.gbm <- rbind(pred.data2.gbm, pred.data.gbm[73816:135483,])

saveRDS(pred.data1.xgb, file = 'pred.data1.xgb.rds')
saveRDS(pred.data2.xgb, file = 'pred.data2.xgb.rds')
saveRDS(pred.data1.rf, file = 'pred.data1.rf.rds')
saveRDS(pred.data2.rf, file = 'pred.data2.rf.rds')
saveRDS(pred.data1.gbm, file = 'pred.data1.gbm.rds')
saveRDS(pred.data2.gbm, file = 'pred.data2.gbm.rds')

t <- pred.data1.xgb[1:73815, ]
t <- t(apply(t, 1, function(x) x/sum(x)))
pred.data1.xgb[1:73815, ] <- t
