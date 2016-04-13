setwd("~/Dropbox/airbnb")

# read data
train <- read.csv("train_users_2.csv")
test <- read.csv("test_users.csv")
sessions <- read.csv("sessions.csv")
bkts <- read.csv("age_gender_bkts.csv")
countries <- read.csv("countries.csv")
sub.sample <- read.csv("sample_submission_NDF.csv")

for (i in 5:15) {
  print(colnames(train)[i])
  print(table(train[,i]))
  print("=================================")
}

countries <- names(sort(table(labels), decreasing = TRUE))[1:5]
submission <- data.frame()

# ============= assign top 5 destinations as default submission ============== 
for (i in 1:dim(sub.sample)[1]) {
  for (j in 1:5) {
    submission[(i-1)*5+j,2] <- countries[j]
  }
  print(i)
}

colnames(submission) <- c("id", "country")
write.csv(submission, file="sub1.csv", row.names=FALSE)

# ================= organize/extract sessions ================
sessions[is.na(sessions)] <- 0

actions <- levels(sessions[,4])
features <- matrix(numeric(length(actions)), 1, length(actions))
colnames(features) <- actions
previous <- as.character(sessions[,1][1])
id <- previous
index <- 1
count <- 0
num.actions <- c()

tic <- Sys.time()

size <- dim(sessions)[1]
for (i in 1:size) {
  current <- as.character(sessions[,1][i])
  
  if (current != previous) {
    id <- c(id, current)
    num.actions <- c(num.actions, count)
    count <- 1
    index <- index + 1
    features <- rbind(features, numeric(length(actions)))
  } else {
    count <- count + 1
  } 
  
  if (i == size) num.actions <- c(num.actions, count)
    
  index.temp <- which(actions == as.character(sessions$action_detail[i]))
  value.temp <- as.numeric(sessions$secs_elapsed[i])
  entry.temp <- features[index, index.temp] + value.temp
  features[index, index.temp] <- entry.temp
  
  previous <- current
}

Sys.time() - tic 

# feature data frame: id + number of actions + time elapsed for each action
df.features <- cbind(id, num.actions, features)
df.features <- as.data.frame(df.features)
df.features <- df.features[df.features$id != '', ]
write.csv(df.features, file = 'features.csv', row.names = FALSE)

# get features of training ids
id.train <- as.vector(intersect(df.features$id, train$id))
labels.train <- c()
ind.train <- c()

size <- length(id.train)
for (i in 1:size) {
  id.temp <- id.train[i]
  ind1.temp <- which(df.features$id == id.temp)
  ind2.temp <- which(train$id == id.temp)
  label.temp <- as.character(train$country_destination[ind2.temp])
  
  ind.train[i] <- ind1.temp
  labels.train[i] <- label.temp
  print(i)
}

train.features <- df.features[ind.train, ]
labels.train <- as.factor(labels.train)

# get features of test ids in sessions
id.test <- as.vector(intersect(df.features$id, test$id))
ind.test <- c()

tic <- Sys.time()

size <- length(id.test)
for (i in 1:size) {
  id.temp <- id.test[i]
  ind.temp <- which(df.features$id == id.temp)
  ind.test[i] <- ind.temp
  print(i)
}

Sys.time() - tic

test.features <- df.features[ind.test, ]

# ================== svm ==================
library(e1071)

tic <- Sys.time()
svm.model <- svm(train.features[,-1], labels.train, cost = 2000, 
                 gamma = 0.0001, cachesize = 5000)
Sys.time() - tic

# predict
tic <- Sys.time()
fit.test <- predict(svm.model, test.features[,-1])
Sys.time() - tic

table(fit.test)

# generate submission
sub1 <- read.csv('sub1.csv')
top5 <- as.vector(sub1[1:5, 2])

size <- dim(test)[1]
for (i in 1:size) {
  id.temp <- as.character(test$id[i])
  if (id.temp %in% id.test) {
    ind.temp <- which(id.test == id.temp)
    top.temp <- fit.test[ind.temp]
    pred.temp <- c(top.temp, top5[top5 != top.temp][1:4])
    sub1[(i*5-4):(i*5), 2] <- pred.temp
  }
}

# ================= binary svm ===================
labels.bi <- as.factor(as.numeric(labels.train=='NDF'))

tic <- Sys.time()
svm.bi.model <- svm(train.features[,-1], labels.bi, cost = 0.1, gamma = 0.0000000001)
Sys.time() - tic

# predict
tic <- Sys.time()
fit.bi.test <- predict(svm.bi.model, test.features[,-1])
Sys.time() - tic

table(fit.bi.test)

# =================== random forest ====================
library(randomForest)

tic <- Sys.time()
rf.model <- randomForest(train.features[,-1], labels.bi, importance = TRUE, ntree = 2000)
Sys.time() - tic

tic <- Sys.time()
fit.rf.test <- predict.randomForest(rf.model, test.features[,-1], OOB=TRUE, type = "response")
Sys.time() - tic

# multicore
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

# ================= test set tweak ==================
id.ses.te <- numeric(id.test) 
tic <- Sys.time()
for(i in 1:length(id.test)) {
  id.ses.te <- id.test[i] %in% id.sessions
  if (i %% 1000 == 0) {
    print(i)
    print(Sys.time() - tic)
  }
}

labels.ses.tr <- labels[which(id.ses.tr==1)]

