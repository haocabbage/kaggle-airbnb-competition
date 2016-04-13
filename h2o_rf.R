library(h2o)
h2o.init(nthreads = -1, max_mem_size = "40G")
h2o.removeAll()

# ============ data loading ===============
features.xgb <- readRDS('features.xgb.rds')
labels <- readRDS('labels.rds')

X <- as.data.frame(cbind(labels, features.xgb[1:73815, -1]))
X.h2o <- as.h2o(X, destination_frame = 'X')

splits <- h2o.splitFrame(X.h2o, ratios = c(0.2, 0.2, 0.2, 0.2), seed = 42)
tr.temp <- h2o.assign(h2o.rbind(splits[[1]], splits[[2]], 
                                splits[[3]], splits[[4]]), 'tr.temp')
va.temp <- h2o.assign(splits[[5]], 'va.temp')

data <- read.csv('data.csv')
data.log <- read.csv('data_log.csv')

data.tr <- as.data.frame(cbind(labels, data[1:73815, ]))
data.log.tr <- as.data.frame(cbind(labels, data.log[1:73815, ]))

data.tr.h2o <- as.h2o(data.tr, destination_frame = 'data.tr.h2o')
data.log.tr.h2o <- as.h2o(data.log.tr, destination_frame = 'data.log.tr.h2o')

splits <- h2o.splitFrame(data.tr.h2o, ratios = c(0.4, 0.4), seed = 42)
data.tr.temp <- h2o.assign(h2o.rbind(splits[[1]], splits[[2]]), 'data.tr.temp')
data.va.temp <- h2o.assign(splits[[3]], 'data.va.temp')

splits <- h2o.splitFrame(data.log.tr.h2o, ratios = c(0.4, 0.4), seed = 42)
data.log.tr.temp <- h2o.assign(h2o.rbind(splits[[1]], splits[[2]]), 'data.log.tr.temp')
data.log.va.temp <- h2o.assign(splits[[3]], 'data.log.va.temp')
#===========================================

tic <- Sys.time()
rf.temp <- h2o.randomForest(training_frame = tr.temp, 
                            validation_frame = va.temp,
                            x = 2:610,                       
                            y = 1,                         
                            mtries = 80, 
                            ntrees = 200, 
                            balance_classes = FALSE,
                            max_depth = 40,               
                            stopping_rounds = 5,          
                            stopping_tolerance = 1e-2,
                            score_each_iteration = T,     
                            seed = 42) 
Sys.time() - tic

tic <- Sys.time()
rf.model <- h2o.randomForest(training_frame = X.h2o, 
                            x = 2:610,                       
                            y = 1,                         
                            mtries = 80, 
                            ntrees = 200, 
                            balance_classes = FALSE,
                            max_depth = 40,               
                            stopping_rounds = 5,          
                            stopping_tolerance = 1e-2,
                            score_each_iteration = T,     
                            seed = 42) 
Sys.time() - tic

tic <- Sys.time()
rf.data.model <- h2o.randomForest(training_frame = data.tr.h2o, 
                                  x = 2:314,                       
                                  y = 1,                         
                                  mtries = 20, 
                                  ntrees = 200, 
                                  balance_classes = FALSE,
                                  max_depth = 20,               
                                  stopping_rounds = 5,          
                                  stopping_tolerance = 1e-2,
                                  score_each_iteration = T,     
                                  seed = 42) 
Sys.time() - tic

tic <- Sys.time()
rf.data.log.model <- h2o.randomForest(training_frame = data.log.tr.h2o, 
                            x = 2:314,                       
                            y = 1,                         
                            mtries = 20, 
                            ntrees = 200, 
                            balance_classes = FALSE,
                            max_depth = 20,               
                            stopping_rounds = 5,          
                            stopping_tolerance = 1e-2,
                            score_each_iteration = T,     
                            seed = 42) 
Sys.time() - tic

h2o.shutdown()
