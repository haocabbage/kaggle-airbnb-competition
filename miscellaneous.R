# ============ replace last 3 with 3 of top 5 =============
sub.temp <- read.csv('sub50.csv')
sub.temp[,2] <- as.vector(sub.temp[,2])
top5 <- c('NDF', 'US', 'other', 'FR', 'IT')

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

write.csv(sub.temp, file = 'sub51.csv', row.names = FALSE)

# ============ replace last 4 with 4 of top 5 =============
sub.temp <- read.csv('sub7.csv')
sub.temp[,2] <- as.vector(sub.temp[,2])
top5 <- c('NDF', 'US', 'other', 'FR', 'IT')

tic <- Sys.time()
for (i in 1:dim(test)[1]) {
  id.temp <- as.character(test$id[i])
  if (id.temp %in% id.test) {
    ind.temp <- which(id.test == id.temp)
    cou.temp <- sub.temp[(i*5-4), 2]
    cou1.temp <- top5[top5!=cou.temp]
    sub.temp[(i*5-4):(i*5), 2] <- c(cou.temp, cou1.temp[1:4])
  }
}
Sys.time() - tic

write.csv(sub.temp, file = 'sub10.csv', row.names = FALSE)


# decompose session features into 11 sub classes
features.de <- rbind(features.tr.all, features.ts.all)

aboveMean <- function(v) {
  # decompose based on quantile
  v.new <- v
  if (length(v[v != 0]) > 0) {
    thresh <- mean(v[v != 0])
    for (i in 1:length(v)) {
      if (v[i] == 0) {
        v.new[i] <- 'none'
      } else if (v[i] < thresh) {
        v.new[i] <- 'zero'
      } else {
        v.new[i] <- 'one'
      }
    }
  }
  return(v.new)
}

features.temp <- apply(features.de[,146:301], 2, aboveMean)
features.temp <- data.frame(features.temp)

dummies <- dummyVars("~.", data = features.temp)
features.dum <- data.frame(predict(dummies, newdata = features.temp))  

features.de <- cbind(features.de[,1:145], features.dum)

de.tr.eho <- features.de[1:NROW(features.tr.all),]
de.ts.eho <- features.de[(NROW(features.tr.all) + 1):NROW(features.de),]

# ================== combine sub16 and air's pred =================
submission
sub.temp <- read.csv('sub16.csv')

for (id in test$id) {
  if (!(id %in% id.test)) {
    ind.temp <- which(test$id == id)
    cou.temp <- submission[(i*5-4):(i*5), 2][1:2]
    cou1.temp <- top5[top5!=cou.temp[1] & top5!=cou.temp[2]]
    sub.temp[(i*5-4):(i*5), 2] <- c(cou.temp, cou1.temp[1:3])      
  }
}

write.csv(sub.temp, file = 'sub18.csv', row.names = FALSE)

quantile(v.temp[v.temp != 0], seq(0, 1, 0.1))[1]


# =================== feature visulization ====================
for (i in 146:301) {
  temp <- features.tr.all[,i]
  if (length(temp[temp!=0])) {
    print(i)
    hist(temp[temp!=0])
  }
}

for (lang in unique(train$language)) {
  ind.temp <- which(train$language == lang)
  print(lang)
  print(sort(table(train$country_destination[ind.temp]), 
             decreasing = TRUE))
}

# =================== exp ====================
tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(features.tr.eho), 
                       label = labels.tr.xgb, 
                       eta = 0.02,
                       max_depth = 5,
                       nfold = cv.nfold,
                       nrounds = 1000, 
                       subsample = 0.8,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(features.ohe.tr[,-1]), 
                       label = unclass(labels) - 1, 
                       eta = 0.1,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = cv.nround, 
                       subsample = 0.7,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12,
                       missing = NA)
Sys.time() - tic

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(features.ohe.tr[,-1]), 
                       label = unclass(labels) - 1, 
                       eta = 0.05,
                       max_depth = 8,
                       nfold = cv.nfold,
                       nrounds = 500, 
                       subsample = 0.9,
                       colsample_bytree = 0.8,
                       eval_metric = ndcg5,
                       objective = "multi:softprob",
                       num_class = 12,
                       missing = NA)
Sys.time() - tic

tic <- Sys.time()
d <- dist2me(features.dist[1,], features.dist.no.NA, 0.2)
Sys.time() - tic

id.sessions <- unique(sessions$user_id)
ind <- c()

which(features.norm$id == as.character(id.sessions[1]))

for (i in 137022:SIZE) {
  id.temp <- as.character(features.norm$id[i])
  if (id.temp %in% id.sessions) {
    ind <- c(ind, i)
    print(i)
  }
}

features.sessions <- features.norm[ind, c(1, 14:3769)]
write.csv(features.sessions, 'features_sessions.csv', row.names = FALSE)

hasNA <- function(v) {
  ind <- which(is.na(v))
  if (length(ind) > 0) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

tic <- Sys.time()
ind.NA <- which(apply(features.sessions, 1, hasNA))
Sys.time() - tic

features.to.fill <- apply(features.sessions[,2:3757], 1, as.numeric)
features.to.fill <- t(features.to.fill)
features.NA <- features.to.fill[ind.NA,]
features.no.NA <- features.to.fill[-ind.NA,]
features.dist <- rbind(features.NA, features.no.NA)
saveRDS(ind.NA, file = 'ind.NA.rds')
saveRDS(features.dist, file = 'features.dist.rds')
saveRDS(features.to.fill, file = 'features.to.fill.rds')

# ======================================================================
features.complete <- readRDS('features.complete.rds')
ind.sessions <- readRDS('ind.sessions.rds')
features.imputed <- readRDS('features.imputed.rds')
features.sessions <- readRDS('features.sessions.rds')

features.complete.imputed <- cbind(features.complete[ind.sessions, 2:145], 
                                   features.imputed)

which(features.sessions$id == as.character(test$id[1]))

features.train <- features.complete.imputed[1:73815, ]
features.test <- features.complete.imputed[73816:NROW(features.complete.imputed), ]
id.test <- as.vector(features.sessions$id[73816:NROW(features.complete.imputed)])

labels <- train$country_destination[ind.sessions[1:73815]]
labels.num <- unclass(labels)-1

library(xgboost)

tic <- Sys.time()
xgb.model <- xgb.cv(data = data.matrix(features.train), 
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
xgb.model <- xgboost(data = data.matrix(features.train), 
                     label = labels.num, 
                     eta = 0.05,
                     max_depth = 8,
                     nrounds = 113, 
                     subsample = 0.9,
                     colsample_bytree = 0.8,
                     eval_metric = ndcg5,
                     objective = "multi:softprob",
                     num_class = 12,
                     missing = NA)
Sys.time() - tic

tic <- Sys.time()
xgb.fit <- predict(xgb.model, data.matrix(features.test), missing = NA)
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

write.csv(sub.xgb.raw, file = 'sub35.csv', row.names = FALSE)

# ====================== retreive de.tr.eho ============================
features.temp <- apply(df.features[,3:158], 2, aboveMean)
features.temp <- data.frame(features.temp)

dummies <- dummyVars("~.", data = features.temp)
features.dum <- data.frame(predict(dummies, newdata = features.temp))  

age <- features.complete[,2]
which(colnames(features.complete) == 'unnamed.mean') #303
de.eho <- cbind(features.complete[,1:302], features.dum)
de.tr.eho <- de.eho[1:73815,]
de.ts.eho <- de.eho[73816:NROW(de.eho),]

tic <- Sys.time()
xgb.model <- xgb.cv(data = data.matrix(de.tr.eho[,-1]), 
                    label = labels.num, 
                    eta = 0.05,
                    max_depth = 8,
                    nfold = 5,
                    nrounds = 500, 
                    subsample = 0.9,
                    colsample_bytree = 0.8,
                    eval_metric = ndcg5,
                    objective = "multi:softprob",
                    num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.model <- xgboost(data = data.matrix(de.tr.eho[,-1]), 
                     label = labels.num, 
                     eta = 0.05,
                     max_depth = 8,
                     nrounds = 112, 
                     subsample = 0.9,
                     colsample_bytree = 0.8,
                     eval_metric = ndcg5,
                     objective = "multi:softprob",
                     num_class = 12)
Sys.time() - tic

tic <- Sys.time()
xgb.fit <- predict(xgb.model, data.matrix(de.ts.eho[,-1]))
Sys.time() - tic

id.test <- as.vector(features.complete$id[73816:NROW(features.complete)])

