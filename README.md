library(tidyverse)
library(naniar)
library(UpSetR)
library(stringr)
library(ggplot2)
library(stats)
library(anchors)
library(corrplot)
library(randomForest)

house = read.csv("house_train.csv")

#Understanding the data
head(house)

#Let's visualize the spread of house prices
ggplot(house, aes(x=SalePrice)) + geom_histogram()

#Visualise missing values
vis_miss(house[,1:30])
#We can see that the 'Alley' column is missing for 93% of the data

vis_miss(house[,51:81])
#Fireplace, PoolQC, Fence, and MiscFeature also have many NA values. 

#We can also get a sense of connections of records with missing values
gg_miss_upset(house)
#471 records have missing values for all 5 of these columns
#Let's get a sense of these columns
house%>%count(PoolArea)
house%>%count(PoolQC)
#We can see that there are 1453 NAs for PoolQC and 1453 values where PoolArea = 0. Therefore, we 
#can assume that PoolQC is NA when there is no pool. We can replace NA with the string "NA" here
house$PoolQC = str_replace_na(house$PoolQC)

house%>%count(Fireplaces)
house%>%count(FireplaceQu)
#We see a similar occurrence for FireplaceQu. Let's replace al NA in FireplaceQu with the string NA
house$FireplaceQu = str_replace_na(house$FireplaceQu)

house$MiscFeature = str_replace_na(house$MiscFeature)
house$Fence = str_replace_na(house$Fence)

house%>%count(Alley)
#Alley has too few non-NAs to be meaningful so we just discard this feature
house$Alley = NULL

#Looking at Lot_Frontage, we jus change NA to 0; maybe these houses do not actually have a front of sorts
house$LotFrontage[is.na(house$LotFrontage)] = 0

#All the garage-reated NA values point to the same records; we can simply assume that these
#places have no garage
house$GarageType = str_replace_na(house$GarageType)
house$GarageYrBlt[is.na(house$GarageYrBlt)] = 0
house$GarageFinish = str_replace_na(house$GarageFinish)
house$GarageQual = str_replace_na(house$GarageQual)
house$GarageCond = str_replace_na(house$GarageCond)

#ikewise, all the basement-related NAs records point to houses without a basement
house%>%count(BsmtFinType1)
house$BsmtCond = str_replace_na(house$BsmtCond)
house$BsmtQual = str_replace_na(house$BsmtQual)
house$BsmtExposure = str_replace_na(house$BsmtExposure)
house$BsmtFinType1 = str_replace_na(house$BsmtFinType1)
house$BsmtFinType2 = str_replace_na(house$BsmtFinType2)
house$BsmtFinSF1 = str_replace_na(house$BsmtFinSF1)
house$BsmtFinSF2 = str_replace_na(house$BsmtFinSF2)

house$MasVnrType = str_replace_na(house$MasVnrType)
house$MasVnrArea[is.na(house$MasVnrArea)] = 0
house$Electrical = str_replace_na(house$Electrical)

#All missing values have been taken care of. Let us now explore the data
head(house)
summary(house)
#We can get rid of the ID column, it has no purpose
house$Id = NULL

#Our target variable, house price, is visualized below
ggplot(house, aes(x=SalePrice)) + geom_histogram()
#Most houses fall in the $200,000 range; there is positive skewness

#Assuming this data is from 2011, we will change the house built and garage date to number of years
house$GarageYrBlt = 2011 - house$GarageYrBlt
house$YearBuilt = 2011 - house$YearBuilt
house$YearRemodAdd = 2011 - house$YearRemodAdd

#Replace the 2011 in GarageYrBuilt to 0
house = replace.value(house, "GarageYrBlt", 2011, 0)

#Let us develop clusters of houses with respect to its rooms information. We want to see how house 
#types vary with respect to their age, lot area, total basement area, living room area, and garage area.
house_cluster = house[,c(4,18,37,61,45)]

head(house_cluster)
#Now we scale the values
house_scaled <- scale(house_cluster)
apply(house_scaled,2,sd) 
#we should get sd=1
apply(house_scaled,2,mean) 
#we should get mean = 0
###kmeans with 4 clusters
house_kmeans <- kmeans(house_scaled,4,nstart=10)
colorcluster <- house_kmeans$cluster
colorcluster

house_kmeans$centers

kIC <- function(fit, rule=c("A","B","C")){
  df <- length(fit$centers) # K*dim
  n <- sum(fit$size)
  D <- fit$tot.withinss # deviance
  rule=match.arg(rule)
  if(rule=="A")
    #return(D + 2*df*n/max(1,n-df-1))
    return(D + 2*df)
  else if(rule=="B") 
    return(D + log(n)*df)
  else 
    return(D +  sqrt( n * log(df) )*df)
}

###computing # of clusters in our example:
kfit <- lapply(1:30, function(k) kmeans(house_scaled,k,nstart=5))
#choose number of clusters based on the fit above
#we will use the  script kIC in DataAnalyticsFunctions.R
#We call the function kIC the performance of the various 
#kmeans for k=1,...50, that was stored in kfit.
#Then "A" for AICc (default) or "B" for BIC
kaic <- sapply(kfit, kIC)
kbic  <- sapply(kfit,kIC,"B")
kHDic  <- sapply(kfit,kIC,"C")
##Now we plot them, first we plot AIC
par(mar=c(1,1,1,1))
par(mai=c(1,1,1,1))
plot(kaic, xlab="k (# of clusters)", ylab="IC (Deviance + Penalty)", 
     ylim=range(c(kaic,kbic,kHDic)), # get them on same page
     type="l", lwd=2)
#Vertical line where AIC is minimized
abline(v=which.min(kaic))
#Next we plot BIC
lines(kbic, col=4, lwd=2)
#Vertical line where BIC is minimized
abline(v=which.min(kbic),col=4)
#Next we plot HDIC
lines(kHDic, col=3, lwd=2)
#Vertical line where HDIC is minimized
abline(v=which.min(kHDic),col=3)

#Insert labels
text(c(which.min(kaic),which.min(kbic),which.min(kHDic)),c(mean(kaic),mean(kbic),mean(kHDic)),c("AIC","BIC","HDIC"))
#both AICc and BIC choose more complicated models
#We choose 4 clusters as the optimal number, as done previously
house_cluster = data.frame(colorcluster)
house = cbind(house, house_cluster)
head(house)
house$colorcluster = as.factor(house$colorcluster)

#Let us perform a PCA on the data to identify which variables are the most useful for us
library(plfm)
nums <- unlist(lapply(house, is.numeric))
pca.house <- prcomp(house[,nums], scale=TRUE)
plot(pca.house,main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors",  line=1, font=2)
#We just select the first three factors
housepc <- predict(pca.house)
housepc
plot(housepc[,1:2], pch=21,  main="")
text(housepc[,1:2], col="blue", cex=1)

#Let's see the most important variables in the first 3 factors
loadings <- pca.house$rotation[,1:3]

v<-loadings[order(abs(loadings[,1]), decreasing=TRUE)[1:27],1]
loadingfit <- lapply(1:27, function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]

v<-loadings[order(abs(loadings[,2]), decreasing=TRUE)[1:27],2]
loadingfit <- lapply(1:27, function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]

v<-loadings[order(abs(loadings[,3]), decreasing=TRUE)[1:27],3]
loadingfit <- lapply(1:27, function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]

#Let's finally run a corrplot
house_cor = cor(house[,nums])
corrplot(house_cor)

house$BsmtFinSF1 = as.integer(house$BsmtFinSF1)
house$BsmtFinSF2 = as.integer(house$BsmtFinSF2)
house$MiscFeature = NULL
house$Electrical = NULL

#We can start running some models on the data. We have to be careful in variable 
#selection as many variables are unlikely to be needed. We will select the variables which form the PCAs
lm1 = lm(SalePrice~(colorcluster)+OverallQual+FullBath+BsmtFullBath+MSSubClass+BsmtQual+ExterQual+KitchenQual+X1stFlrSF+TotalBsmtSF+GarageCars, house)
summary(lm1)

#Tree model
library(rpart)
library(rpart.plot)
set.seed(123)
tree_model = rpart(house$SalePrice ~ ., data = house, method = "anova")
tree_model$variable.importance
rpart.plot(tree_model)

#Random Forest
set.seed(123)
sample_random <- randomForest((SalePrice)~GrLivArea+X1stFlrSF+colorcluster+MSSubClass+GarageArea+ExterQual+Neighborhood+OverallQual+FullBath+TotalBsmtSF+YearBuilt, data=house, importance=T, ntree=500)
sample_random$importance
sample_random$predicted
print(sample_random)

#OOS
set.seed(123)
nfold <- 10
n <- nrow(house) # the number of observations
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
model.reg <- lm(SalePrice~(colorcluster)+OverallQual+FullBath+BsmtFullBath+MSSubClass+BsmtQual+ExterQual+KitchenQual+X1stFlrSF+TotalBsmtSF+GarageArea+GrLivArea, house, subset=which(foldid==1))
model.reg
model.null <- glm(SalePrice~1, data=house, subset=which(foldid==1))
model.tree <- rpart(SalePrice ~ ., data=house, method="anova", subset=which(foldid==1))
model.rf <- randomForest((SalePrice)~GrLivArea+X1stFlrSF+colorcluster+MSSubClass+GarageArea+ExterQual+Neighborhood+OverallQual+FullBath+TotalBsmtSF+YearBuilt, data=house, importance=T, ntree=500, subset=which(foldid==1))

R2 <- function(y, pred, family=c("gaussian","binomial")){
  fam <- match.arg(family)
  if(fam=="binomial"){
    if(is.factor(y)){ y <- as.numeric(y)>1 }
  }
  dev <- deviance(y, pred, family=fam)
  dev0 <- deviance(y, mean(y), family=fam)
  return(1-dev/dev0)
}

R2.rf <-R2(y=house$SalePrice, pred=predict(sample_random, newdata=house))
R2.rf

OOS <- data.frame(rf=rep(NA, nfold), reg=rep(NA, nfold), tree=rep(NA, nfold), null=rep(NA, nfold)) 
house = data.frame(house)
#Set the other part for training (if not k)
traink <- which(foldid!=k) # train on all but fold `k'
testk  <- which(foldid==k) # test on fold k

pred.reg <- predict(model.reg, newdata=house[-traink,])
pred.tree   <- predict(model.tree, newdata=house[-traink,], type="vector")
pred.null <- predict(model.null, newdata=house[-traink,])
pred.rf = predict(model.rf, newdata=house[-traink,])
nfold
###Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  traink <- which(foldid!=k) # train on all but fold `k'
  
  ##fit the two regressions and null model
  model.rf <- randomForest((SalePrice)~GrLivArea+X1stFlrSF+colorcluster+MSSubClass+GarageArea+ExterQual+Neighborhood+OverallQual+FullBath+TotalBsmtSF+YearBuilt, data=house, importance=T, ntree=500, subset=traink)
  model.reg <- lm(SalePrice~(colorcluster)+OverallQual+FullBath+BsmtFullBath+MSSubClass+BsmtQual+ExterQual+KitchenQual+X1stFlrSF+TotalBsmtSF+GarageCars, house, subset=traink)
  model.tree <- rpart(SalePrice ~ ., data=house, method="anova", subset=traink)
  model.nulll <-glm(SalePrice~1, data=house, subset=traink)
  ##get predictions: type=response so we have probabilities
  pred.rf = predict(model.rf, newdata=house[-traink,])
  pred.reg             <- predict(model.reg, newdata=house[-traink,])
  pred.tree                 <- predict(model.tree, newdata=house[-traink,])
  pred.null <- predict(model.null, newdata=house[-traink,])
  ##calculate and log R2
  #RF
  OOS$rf[k] <- R2(y=house$SalePrice[-traink], pred=pred.rf)
  OOS$rf[k]
  #Reg
  OOS$reg[k] <- R2(y=house$SalePrice[-traink], pred=pred.reg)
  OOS$reg[k]
  #Tree
  OOS$tree[k] <- R2(y=house$SalePrice[-traink], pred=pred.tree)
  OOS$tree[k]
  #Null
  OOS$null[k] <- R2(y=house$SalePrice[-traink], pred=pred.null)
  OOS$null[k]
  #Null Model guess
  sum(house$SalePrice[traink]=="Yes")/length(traink)
  
  ##We will loop this nfold times (I setup for 10)
  ##this will print the progress (iteration that finished)
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

colMeans(OOS)
m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)
barplot(t(as.matrix(OOS)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample " ~ R^2), xlab="Fold", names.arg = c(1:10))

#Working with the test set
house_test = read.csv("house_test.csv")
house_test$GarageType = str_replace_na(house_test$GarageType)
house_test$GarageYrBlt[is.na(house_test$GarageYrBlt)] = 0
house_test$GarageFinish = str_replace_na(house_test$GarageFinish)
house_test$GarageQual = str_replace_na(house_test$GarageQual)
house_test$GarageCond = str_replace_na(house_test$GarageCond)

house_test$GarageYrBlt = 2011 - house_test$GarageYrBlt
house_test$YearBuilt = 2011 - house_test$YearBuilt
house_test$YearRemodAdd = 2011 - house_test$YearRemodAdd
max(house_test$GarageYrBlt)

#Replace the 2011 in GarageYrBuilt to 0
house = replace.value(house, "GarageYrBlt", 2011, 0)
house_test$FireplaceQu = str_replace_na(house_test$FireplaceQu)
house_test$Alley = NULL
house_test$MiscFeature = str_replace_na(house_test$MiscFeature)
house_test$Fence = str_replace_na(house_test$Fence)

house_test$LotFrontage[is.na(house_test$LotFrontage)] = 0

house_test$BsmtCond = str_replace_na(house_test$BsmtCond)
house_test$BsmtQual = str_replace_na(house_test$BsmtQual)
house_test$BsmtExposure = str_replace_na(house_test$BsmtExposure)
house_test$BsmtFinType1 = str_replace_na(house_test$BsmtFinType1)
house_test$BsmtFinType2 = str_replace_na(house_test$BsmtFinType2)
house_test$BsmtFinSF1 = as.integer(str_replace_na(house_test$BsmtFinSF1))
house_test$BsmtFinSF2 = as.integer(str_replace_na(house_test$BsmtFinSF2))

house_test$MasVnrType = str_replace_na(house_test$MasVnrType)
house_test$MasVnrArea[is.na(house_test$MasVnrArea)] = 0
house_test$Electrical = str_replace_na(house_test$Electrical)

house_test$PoolQC = str_replace_na(house_test$PoolQC)

house_test$BsmtHalfBath[is.na(house_test$BsmtHalfBath)] = 0
house_test$BsmtFullBath[is.na(house_test$BsmtFullBath)] = 0
house_test$GarageArea[is.na(house_test$GarageArea)] = 0
house_test$GarageCars[is.na(house_test$GarageCars)] = 0
house_test$KitchenQual[is.na(house_test$KitchenQual)] = "Gd"
house_test$TotalBsmtSF[is.na(house_test$TotalBsmtSF)] = 0
house_test$SaleType = NULL
house_test$Utilities = NULL
house_test$Functional = NULL
house_test$MSZoning = NULL
house_test$BsmtFinSF1[is.na(house_test$BsmtFinSF1)] = 0
house_test$BsmtFinSF2[is.na(house_test$BsmtFinSF2)] = 0
house_test$Exterior1st = NULL
house_test$Exterior2nd = NULL
house_test$BsmtUnfSF[is.na(house_test$BsmtUnfSF)] = 0

testcluster = house_test[,c("LotArea", "YearBuilt", "TotalBsmtSF", "GarageArea", "GrLivArea")]
house_test_scaled <- scale(testcluster)
apply(house_test_scaled,2,sd) 
#we should get sd=1
apply(house_test_scaled,2,mean) 
#we should get mean = 0
###kmeans with 4 clusters
house_test_kmeans <- kmeans(house_test_scaled,4,nstart=10)
colorcluster <- house_test_kmeans$cluster
colorcluster
house_test$colorcluster = as.factor(colorcluster)

predictions = predict(sample_random, newdata=house_test)
predictions = data.frame(predictions)
