evaluate <- function(preds, labels) {
  num.class = 12
  pred <- t(preds)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log2(2:(length(y)+1)))
  ndcg <- mean(apply(x,1,dcg))
  return(ndcg)
}

xgb1 <- t(pred.data1.xgb[1:73815,])
rf1 <- t(pred.data1.rf[1:73815,])
gbm1 <- t(pred.data1.gbm[1:73815,])

xgb2 <- t(pred.data2.xgb[1:73815,])
rf2 <- t(pred.data2.rf[1:73815,])
gbm2 <- t(pred.data2.gbm[1:73815,])

evaluate(t(xgb1), labels.num)
evaluate(t(xgb2), labels.num)
evaluate(t(rf1), labels.num)
evaluate(t(rf2), labels.num)
evaluate(t(gbm1), labels.num)
evaluate(t(gbm2), labels.num)

xgb.aver <- as.vector(t(pred.data1.xgb[73816:135483, ]))
rf.aver <- as.vector(t(pred.data1.rf[73816:135483, ]))

m <- cbind(xgb.aver, rf.aver)

# arithmetic mean 
v.ari <- apply(m, 1, function(x) sum(x*c(0.6, 0.4))) 
# geometric mean
v.geo <- apply(m, 1, function(x) prod(x^c(0.8, 0.2)))
# harmonic mean
v.har <- apply(m, 1, function(x) 1/sum(c(0.9, 0.1)/x))

m.ari <- t(matrix(v.ari, 12))
m.geo <- t(matrix(v.geo, 12))
m.har <- t(matrix(v.har, 12))

evaluate(m.ari, labels.num)
evaluate(m.geo, labels.num)
evaluate(m.har, labels.num)

xgb.fit <- v.geo
xgb.fit <- v.har
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

write.csv(sub.xgb.raw, file = 'sub50.csv', row.names = FALSE)

