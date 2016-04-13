xgb.h2o <- as.h2o(features.xgb[,-1], destination_frame = 'xgb')

pred.rf <- h2o.predict(rf.model, xgb.h2o)
pred.gbm <- h2o.predict(gbm.model, xgb.h2o)

pred.rf <- as.data.frame(pred.rf)[,c(2:10, 13, 11, 12)]
pred.gbm <- as.data.frame(pred.gbm)[,c(2:10, 13, 11, 12)]

pred.rf <- t(apply(pred.rf, 1, function(x) (x + 1)/sum(x + 1)))
pred.gbm <- t(apply(pred.gbm, 1, function(x) (x + 1)/sum(x + 1)))

saveRDS(pred.rf, file = 'pred.rf.rds')
saveRDS(pred.gbm, file = 'pred.gbm.rds')

# =============================================================
data.h2o <- as.h2o(data, destination_frame = 'data.h2o')

pred.data.rf <- h2o.predict(rf.data.model, data.h2o)
pred.data.gbm <- h2o.predict(gbm.data.model, data.h2o)

pred.data.rf <- as.data.frame(pred.data.rf)[,c(2:10, 13, 11, 12)]
pred.data.gbm <- as.data.frame(pred.data.gbm)[,c(2:10, 13, 11, 12)]

pred.data.rf <- t(apply(pred.data.rf, 1, function(x) (x + 1)/sum(x + 1)))
pred.data.gbm <- t(apply(pred.data.gbm, 1, function(x) (x + 1)/sum(x + 1)))

saveRDS(pred.data.rf, file = 'pred.data.rf.rds')
saveRDS(pred.data.gbm, file = 'pred.data.gbm.rds')

# =============================================================
data.log.h2o <- as.h2o(data.log, destination_frame = 'data.log.h2o')

pred.data.log.rf <- h2o.predict(rf.data.log.model, data.log.h2o)
pred.data.log.gbm <- h2o.predict(gbm.data.log.model, data.log.h2o)

pred.data.log.rf <- as.data.frame(pred.data.log.rf)[,c(2:10, 13, 11, 12)]
pred.data.log.gbm <- as.data.frame(pred.data.log.gbm)[,c(2:10, 13, 11, 12)]

pred.data.log.rf <- t(apply(pred.data.log.rf, 1, function(x) (x + 1)/sum(x + 1)))
pred.data.log.gbm <- t(apply(pred.data.log.gbm, 1, function(x) (x + 1)/sum(x + 1)))

saveRDS(pred.data.log.rf, file = 'pred.data.log.rf.rds')
saveRDS(pred.data.log.gbm, file = 'pred.data.log.gbm.rds')