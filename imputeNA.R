matrix.dist <- readRDS('matrix.dist.rds')
features.dist <- readRDS('features.dist.rds')
SIZE <- NROW(features.dist)
ind.NA <- readRDS('ind.NA.rds')
features.dist.NA <- features.dist[1:length(ind.NA), ]
features.dist.no.NA <- features.dist[(length(ind.NA)+1):SIZE, ]

imputeNA <- function(v, m, neighbors) {
  v.imputed <- v
  # get missing indices
  ind.NA <- which(is.na(v))
  # impute NA
  for (i in ind.NA) {
    impute <- median(m[neighbors, i])
    v.imputed[i] <- impute
  }
  return(v.imputed)
}

features.NA.imputed <- features.dist.NA

for (i in 1:NROW(features.dist.NA)) {
  v.temp <- features.dist.NA[i,]
  nei.temp <- matrix.dist[i,][1:1000]
  impute.temp <- imputeNA(v.temp, features.dist.no.NA, nei.temp)
  features.NA.imputed[i,] <- impute.temp
  print(i)
}

saveRDS(features.NA.imputed, file = 'features.NA.imputed.rds')

features.NA.imputed <- readRDS('features.NA.imputed.rds')
ind.NA <- readRDS('ind.NA.rds')
features.to.fill <- readRDS('features.to.fill.rds')

features.imputed <- features.to.fill 
features.imputed[ind.NA,] <- features.NA.imputed

saveRDS(features.imputed, file = 'features.imputed.rds')