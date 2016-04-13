colnames(train.features)[3] <- 'unnamed'
train.features <- train.features[,-1]

colnames(test.features)[3] <- 'unnamed'
test.features <- test.features[,-1]

colnames(train)
train.temp <- train[,-c(1,2,3,4,6,16)]
test.temp <- test[,-c(1,2,3,4,6,16)]

size.train <- dim(train.temp)[1]
tot.temp <- rbind(train.temp, test.temp)

for (i in 1:dim(tot.temp)[2]) {
  tot.temp[,i] <- unclass(tot.temp[,i])
}

train.temp <- tot.temp[1:size.train, ]
test.temp <- tot.temp[(size.train+1):dim(tot.temp)[1], ]

tr.fea.temp <- c()
for (i in 1:length(id.train)) {
  id.temp <- id.train[i]
  ind.temp <- which(train$id == id.temp)
  tr.fea.temp <- rbind(tr.fea.temp, train.temp[ind.temp, ])
  print(i)
}

ts.fea.temp <- c()
for (i in 1:length(id.test)) {
  id.temp <- id.test[i]
  ind.temp <- which(test$id == id.temp)
  ts.fea.temp <- rbind(ts.fea.temp, test.temp[ind.temp, ])
  print(i)
}

features.tr.xgb <- cbind(tr.fea.temp, train.features)
features.ts.xgb <- cbind(ts.fea.temp, test.features)

write.csv(features.tr.xgb, file = 'features_tr_xgb.csv', row.names = FALSE)
write.csv(features.ts.xgb, file = 'features_ts_xgb.csv', row.names = FALSE)

write.csv(id.train, file = 'id_train_sessions.csv', row.names = FALSE)
write.csv(id.test, file = 'id_test_sessions.csv', row.names = FALSE)

classes <- levels(train$country_destination)
labels.tr.xgb <- unclass(labels.train)

write.csv(labels.train, file = 'label_train_sessions.csv', row.names = FALSE)

features <- read.csv('features.csv')
colnames(features)[3] <- 'unnamed'

id.train <- as.vector(read.csv('id_train_sessions.csv')[,1])
id.test <- as.vector(read.csv('id_test_sessions.csv')[,1])
labels.tr <- read.csv('label_train_sessions.csv')[,1]
