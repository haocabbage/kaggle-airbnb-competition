setwd("~/Dropbox/airbnb")
library(xgboost)
library(doMC)
library(iterators)
library(parallel)
library(foreach)

# read data
train <- read.csv("train_users_2.csv")
test <- read.csv("test_users.csv")
sessions <- read.csv("sessions.csv")
labels <- train$country_destination
classes <- levels(labels) 

# extract # of total actions, # of each action detail and average time of 
# each action detail for each id from sessions

# get action detail types
action.detail <- as.vector(unique(sessions$action_detail))

# remove non-identified ids
ind.no_id <- which(sessions$user_id == '')
sessions <- sessions[-ind.no_id, ]

# ============== extract session info ==============
id.train <- as.vector(train$id)
id.test <- as.vector(test$id)

features.sessions <- c()

# set up vectors of new features
id.sessions <- c()
number_of_actions <- c()
number_of_each_action <- vector('list', length(action.detail))
names(number_of_each_action) <- action.detail

action.detail.mean <- action.detail
for (i in 1:length(action.detail)) {
  action.detail.mean[i] <- paste(c(action.detail[i], '.mean'), collapse = '')
}

mean_time_of_each_action <- vector('list', length(action.detail))
names(mean_time_of_each_action) <- action.detail.mean

#initialize
id.current <- as.character(sessions$user_id[1])
id.sessions <- id.current
number_of_actions <- 0

for (j in 1:length(number_of_each_action)) {
  number_of_each_action[[j]] <- 0
  mean_time_of_each_action[[j]] <- 0 
}

len <- 1
num_of_na <- number_of_each_action 

# extract info
for (i in 1:NROW(sessions)) {
  
  id.new <- as.character(sessions$user_id[i]) 
  
  if (id.new != id.current) {
    # initialize new data point
    id.current <- id.new
    id.sessions <- append(id.sessions, id.current)
    number_of_actions <- append(number_of_actions, 0)
    
    for (j in 1:length(number_of_each_action)) {
      number_of_each_action[[j]] <- append(number_of_each_action[[j]], 0)
      num_of_na[[j]] <- append(num_of_na[[j]], 0)
      mean_time_of_each_action[[j]] <- append(mean_time_of_each_action[[j]], 0)
    }
    
    len <- length(id.sessions) 
  }
  
  # update data point info
  number_of_actions[len] <- number_of_actions[len] + 1
  
  action.temp <- as.character(sessions$action_detail[i])
  ind.temp <- which(action.detail == action.temp)
  num.temp <- number_of_each_action[[ind.temp]][len] 
  na.temp <- num_of_na[[ind.temp]][len]
  mean.temp <- mean_time_of_each_action[[ind.temp]][len]
  time.temp <- sessions$secs_elapsed[i]
    
  number_of_each_action[[ind.temp]][len] <- num.temp + 1
  
  if (is.na(time.temp)) {
    num_of_na[[ind.temp]][len] <- na.temp + 1
  } else {
    mean_time_of_each_action[[ind.temp]][len] <- updateMean(mean.temp,
                                                            time.temp,
                                                            num.temp - na.temp) 
  }
}

updateMean <- function(old.mean, new.value, old.num) {
  # update mean
  new.mean <- (old.mean*old.num + new.value)/(old.num + 1)
  return(new.mean)
}

df <- as.data.frame(number_of_each_action)
df1 <- as.data.frame(mean_time_of_each_action)
  
# mark the action with missing mean time for each id
for (i in 1:NCOL(df)) {
  ind.temp <- which(df[,i] > 0 & df1[,i] == 0)
  if (length(ind.temp) > 0) df1[,i][ind.temp] <- NA
}

# combine all session features
features.sessions <- cbind(number_of_actions, df, df1)


# =================== build a complete feature matrix ===================
# raw features with ohe features in train file
features.ohe <- as.data.frame(matrix(NA, NROW(df_all_combined), 
                                      NCOL(df_all_combined) + 
                                       NCOL(features.sessions)))
colnames(features.ohe) <- c(colnames(df_all_combined), 
                            colnames(features.sessions))
features.ohe[,1:NCOL(df_all_combined)] <- df_all_combined

for (i in 1:NROW(features.ohe)) {
  id.temp <- as.character(features.ohe$id[i])
  if (id.temp %in% id.sessions) {
    ind.temp <- which(id.sessions == id.temp)
    replace.temp <- features.sessions[ind.temp, ]
    features.ohe[i, (NCOL(df_all_combined)+1):NCOL(features.ohe)] <- replace.temp   
  }
}

write.csv(features.ohe, file = 'features_complete.csv', row.names = FALSE)

features.ohe.tr <- features.ohe[1:NROW(train), ]
features.ohe.ts <- features.ohe[(NROW(train)+1):NROW(features.ohe),]

# xgb benchmark
cv.nfold <- 2
cv.nround <- 100

tic <- Sys.time()
xgb.model.cv <- xgb.cv(data = data.matrix(features.ohe.tr[,-1]), 
                       label = unclass(labels) - 1, 
                       eta = 0.5,
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

# =================== data value normalization ==================
getCDF <- function(x, v) {
  # get empirical CDF of x based on feature vector v in the training set
  v.sort <- sort(v)
  if (x < v.sort[1]) {
    cdf <- 0
  } else {
    cdf <- length(which(v.sort <= x))/length(v.sort)
  }
  return(cdf)
}

tic <- Sys.time()
features.to.norm <- features.ohe[, c(2, 146:458)]
features.norm <- as.data.frame(features.ohe$id)

for (i in 1:NCOL(features.to.norm)) {
  cat('start feature ')
  cat(i)
  cat('\n')
  # generate column names for the new feature matrix
  name.temp <- colnames(features.to.norm)[i]
  names.new <- c()
  
  # generate 12 sub features
  to.attach <- as.data.frame(matrix(NA, NROW(features.ohe), 12))
  
  for (j in 1:12) {
    names.new <- append(names.new, paste(c(name.temp, '.',classes[j]), 
                                         collapse = ''))    
  }
  
  colnames(to.attach) <- names.new
  
  # assign normalized value
  v <- features.to.norm[,i]
  for (k in 1:12) {
    cat('start class ')
    cat(k)
    cat('\n')
    class.temp <- classes[k]
    ind.temp <- which(labels == class.temp)
    v.sub <- v[ind.temp]
    num.not.na <- length(which(!is.na(v.sub)))
    if (num.not.na > 0) {
      v.sub.not.na <- v.sub[!is.na(v.sub)]
      for (l in 1:NROW(features.ohe)) {
        if (!is.na(v[l])) {
          x.temp <- getCDF(v[l], v.sub.not.na)
          to.attach[l,k] <- x.temp  
        }
      }  
    }
    cat('end class ')
    cat(k)
    cat('\n')
  }
  
  # attach the sub features
  features.norm <- cbind(features.norm, to.attach)
  cat('end feature ')
  cat(i)
  cat('\n')
}
Sys.time() - tic

write.csv(features.norm, file = 'features_complete_norm.csv', row.names = FALSE)

# ================== NA imputation ====================
# impute missing values

