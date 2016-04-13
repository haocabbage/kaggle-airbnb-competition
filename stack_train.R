features.stack <- cbind(pred.data1.xgb,
                        pred.data2.xgb,
                        pred.data1.rf,
                        pred.data2.rf,
                        pred.data1.gbm,
                        pred.data2.gbm)

write.csv(features.stack, file = 'features_stack.csv', row.names = FALSE)

features.stack.tr <- features.stack[1:73815, ]
features.stack.ts <- features.stack[73816:135483, ]
xgb.tr <- features.xgb[1:73815, -1]
xgb.ts <- features.xgb[73816:135483, -1]

tr <- cbind(xgb.tr, features.stack.tr)
  
# xgb
xgb.model.cv <- xgb.cv(data = data.matrix(features.stack.tr), 
                       label = labels.num, 
                       eta = 0.05,
                       max_depth = 8,
                       nfold = 5,
                       nrounds = 100, 
                       subsample = 0.9,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)

xgb.model.cv <- xgb.cv(data = data.matrix(tr), 
                       label = labels.num, 
                       eta = 0.02,
                       max_depth = 8,
                       nfold = 5,
                       nrounds = 250, 
                       subsample = 0.9,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)

xgb.stack.model <- xgboost(data = data.matrix(features.stack.tr), 
                           label = labels.num, 
                           eta = 0.05,
                           max_depth = 8,
                           nrounds = 45, 
                           subsample = 0.8,
                           colsample_bytree = 0.8,
                           eval_metric = ndcg5,
                           objective = "multi:softprob",
                           num_class = 12)

pred.stack.xgb  <- predict(xgb.stack.model, features.stack.ts)
xgb.fit <- pred.stack.xgb 

# =============================================================
stack.tr.h2o <- as.h2o(cbind(labels, features.stack.tr), destination_frame = 'stack.tr.h2o')
stack.ts.h2o <- as.h2o(features.stack.ts, destination_frame = 'stack.ts.h2o')

splits <- h2o.splitFrame(stack.tr.h2o, ratios = c(0.4, 0.4), seed = 42)
stack.tr.temp <- h2o.assign(h2o.rbind(splits[[1]], splits[[2]]), 'stack.tr.temp')
stack.va.temp <- h2o.assign(splits[[3]], 'stack.va.temp')

rf.stack.model <- h2o.randomForest(training_frame = stack.tr.h2o, 
                            x = 2:73,                       
                            y = 1,                         
                            mtries = 10, 
                            ntrees = 200, 
                            balance_classes = FALSE,
                            max_depth = 8,               
                            stopping_rounds = 5,          
                            stopping_tolerance = 1e-2,
                            score_each_iteration = T,     
                            seed = 42) 

pred.stack.rf <- h2o.predict(rf.stack.model, stack.ts.h2o)
pred.stack.rf <- as.data.frame(pred.stack.rf)[,c(2:10, 13, 11, 12)]
pred.stack.rf <- t(apply(pred.stack.rf, 1, function(x) (x + 1)/sum(x + 1)))
saveRDS(pred.stack.rf, file = 'pred.stack.rf.rds')


# generate submission ===========================================
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

write.csv(sub.xgb.raw, file = 'sub46.csv', row.names = FALSE)