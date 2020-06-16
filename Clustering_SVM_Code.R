#------------------------------------------
############## FINAL PROJECT ##############
#------------------------------------------

# Clear workspace
rm(list=ls())

# Load the .RData file 
load("Clustering_SVM_Workspace.RData")

# Load Libraries
library(ggplot2)
library(cluster) #clustering
library(factoextra) #cluster validation
library(fpc) #kmeans cluster plots
library(gridExtra) #plotting
library(caret) #machine learning (feature selection)
library(e1071) # SVM
library(DMwR) #Resampling
library(dplyr)  #select_if function

# Load data
data <- read.csv("weatherAUS.csv")

# View structure and summary
str(data)
summary(data)

# Format date for Date variable
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")
range(data$Date)

# Correlation
cor(select_if(data, is.numeric), use = 'complete')

#------------------------------------------
############## CLUSTERING ##############

# Subset data for 1 year
weather <- subset(data, Date >= "2016-07-01")
range(weather$Date)
str(weather)
summary(weather)

# Irrelevant, redundant features
# Remove highly correlated variables - Temp9am, Temp3pm, Pressure9am
# Correlation
weather_cluster = weather[,-c(16,20,21)]

# Corr between categorical variables
# H0: The two variables are not dependent
# H1: The two variables are dependent
chisq.test(table(weather_cluster$WindGustDir, weather_cluster$WindDir9am))
chisq.test(table(weather_cluster$WindGustDir, weather_cluster$WindDir3pm))
chisq.test(table(weather_cluster$WindDir3pm, weather_cluster$WindDir9am))

# Remove WindDir9am, WinDir3pm
weather_cluster = weather_cluster[,-c(10,11)]

# Remove Date, RainToday, RainTomorrow, WindGustDir, Risk_mm
weather_cluster = weather_cluster[,-c(1,8,17,18,19)]

# Missing values
# Remove 4 columns with lots of missing values
weather_cluster = weather_cluster[,-c(5,6,13,14)]

# Remove rows with missing values
nrow(weather_cluster[!complete.cases(weather_cluster),])
weather_cluster <- na.omit(weather_cluster)
summary(weather_cluster)

# Aggregation by mean values
weather_cluster <- aggregate(weather_cluster[2:10], by=list(weather_cluster$Location), FUN=mean, na.rm=TRUE)
rownames(weather_cluster)=weather_cluster[,1]
weather_cluster=weather_cluster[,-1]
head(weather_cluster)

# Check outliers using Zscore
weather_cluster[abs(scale(weather_cluster$MinTemp))>3,]
weather_cluster[abs(scale(weather_cluster$MaxTemp))>3,]
weather_cluster[abs(scale(weather_cluster$Rainfall))>3,] #Darwin
weather_cluster[abs(scale(weather_cluster$WindGustSpeed))>3,]
weather_cluster[abs(scale(weather_cluster$WindSpeed9am))>3,]
weather_cluster[abs(scale(weather_cluster$WindSpeed3pm))>3,]
weather_cluster[abs(scale(weather_cluster$Humidity9am))>3,] #AliceSprings
weather_cluster[abs(scale(weather_cluster$Humidity3pm))>3,]
weather_cluster[abs(scale(weather_cluster$Pressure3pm))>3,] #Darwin

# Scale data
summary(weather_cluster)
diag(var(weather_cluster))

weather_cluster_scale <- scale(weather_cluster)
summary(weather_cluster_scale)
diag(var(weather_cluster_scale))

# Use the fviz_nbclust() function from the factoextra package 
# to plot total within sum of square for k up to 15 
fviz_nbclust(weather_cluster_scale, 
             FUNcluster=kmeans, 
             method="wss", 
             k.max=15)
## Choose k = 4

# K-means clustering
set.seed(831)
kmeans4 <- kmeans(x=weather_cluster_scale[,1:9], centers=4, trace=FALSE, nstart=30)
kmeans4
kmeans4$tot.withinss
kmeans4$size
fviz_cluster(kmeans4, weather_cluster_scale[,1:9])

### Clustering for 2009 ###
# Subset and clean data - same steps as clustering for 2017
weather_cluster_2009 <- subset(data, Date >= "2009-01-01" & Date <= "2009-12-31")
range(weather_cluster_2009$Date)
weather_cluster_2009 = weather_cluster_2009[,-c(1,6:8,10,11,16,18:24)]
weather_cluster_2009 <- na.omit(weather_cluster_2009)
str(weather_cluster_2009)
summary(weather_cluster_2009)
weather_cluster_2009 <- aggregate(weather_cluster_2009[2:10], by=list(weather_cluster_2009$Location), FUN=mean, na.rm=TRUE)
rownames(weather_cluster_2009)=weather_cluster_2009[,1]
weather_cluster_2009=weather_cluster_2009[,-1]
head(weather_cluster_2009)

# Scale data
weather_cluster_2009_scale <- scale(weather_cluster_2009)
summary(weather_cluster_2009_scale)
diag(var(weather_cluster_2009_scale))

# k-means clustering
fviz_nbclust(weather_cluster_2009_scale, 
             FUNcluster=kmeans, 
             method="wss", 
             k.max=15)
## Choose k = 4

set.seed(831)
kmeans4_2009 <- kmeans(x=weather_cluster_2009_scale[,1:9], centers=4, trace=FALSE, nstart=30)
kmeans4_2009$tot.withinss
fviz_cluster(kmeans4_2009, weather_cluster_2009_scale[,1:9])

#-------------------------------------------
############## CLASSIFICATION ##############

# Remove Date, Location, Evaporation, Sunshine, Cloud9am, Cloud3pm, Risk_mm
weather_sub <- weather[,-c(1,2,6,7,18,19,23)]
str(weather_sub)

# Remove missing values
nrow(weather_sub[!complete.cases(weather_sub),])
weather_sub <- na.omit(weather_sub)
summary(weather_sub)

# Convert categorical variables to binary/dummy variables
df1 <- dummyVars(~WindGustDir+WindDir9am+WindDir3pm+RainToday, data=weather_sub,
                 sep="_", fullRank = TRUE)
df2 <- predict(df1, weather_sub)
weather_pred <- data.frame(weather_sub[,-c(4,6,7,16)], df2)
str(weather_pred)

# Distribution of dependent variable, RainTomorrow
barplot(table(weather_pred$RainTomorrow))

# Split Training & Testing set
set.seed(831)
samp <- createDataPartition(weather_pred$RainTomorrow, p=.80, list=FALSE)
train = weather_pred[samp, ] 
test = weather_pred[-samp, ]

# Resampling using SMOTE
set.seed(831)
train_sm <- SMOTE(RainTomorrow~., data=train)
barplot(table(train_sm$RainTomorrow)) # Class distribution after oversampling

# Radial Kernel
set.seed(831)
svm_modR <- svm(RainTomorrow~., 
                data=train_sm, 
                method="C-classification", 
                kernel="radial",
                scale=TRUE)

# Training Performance
svm.trainR <- predict(svm_modR, 
                      train_sm[,-13],
                      type="class")
svm.train.accR <- confusionMatrix(svm.trainR, 
                                  train_sm$RainTomorrow, 
                                  mode="prec_recall")

#Testing performance
svm.testR <- predict(svm_modR, 
                     test[,-13], 
                     type="class")
svm.test.accR <- confusionMatrix(svm.testR, 
                                 test$RainTomorrow, 
                                 mode="prec_recall")

cbind(train=svm.train.accR$overall, test=svm.test.accR$overall)
cbind(train=svm.train.accR$byClass, test=svm.test.accR$byClass)

# Tune hyperparameters
set.seed(831)
tRadial=tune(svm, RainTomorrow~., data=train_sm, 
             tunecontrol=tune.control(nrepeat=1,sampling = "cross",cross=5), 
             kernel="radial", scale = TRUE,
             ranges=list(cost=c(.0001,.001,.01,.1,10,100,1000),gamma=c(.0001,.001,.01,.1,.5,1,2,3,4)))

## R processed the function long but was still processing, so I couldn't get the best values.

## Try Feature Selection ##
set.seed(831)
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      number = 10,
                      repeats = 3,
                      verbose = FALSE)

can_rfe <- rfe(x = train[,-13], 
               y = train$RainTomorrow,
               rfeControl = control)

can_rfe
plot(can_rfe, type=c("g", "o"))
can_rfe$optVariables

# New dataset with reduced number of variables
train_fs <- cbind(train$RainTomorrow,train[, colnames(train) %in% can_rfe$optVariables])
test_fs <- cbind(test$RainTomorrow,test[,colnames(test) %in% can_rfe$optVariables])

names(train_fs)[1] <- "RainTomorrow"
names(test_fs)[1] <- "RainTomorrow"

# Resampling using SMOTE
set.seed(831)
train_fs_sm <- SMOTE(RainTomorrow~., data=train_fs)
barplot(table(train_fs_sm$RainTomorrow)) # Class distribution after oversampling

# Radial Kernel
set.seed(831)
svm_modR_fs <- svm(RainTomorrow~., 
                data=train_fs_sm, 
                method="C-classification", 
                kernel="radial",
                scale=TRUE)

# Training Performance
svm.trainR_fs <- predict(svm_modR_fs, 
                      train_fs_sm[,-1],
                      type="class")
svm.train.accR_fs <- confusionMatrix(svm.trainR_fs, 
                                  train_fs_sm$RainTomorrow, 
                                  mode="prec_recall")

#Testing performance
svm.testR_fs <- predict(svm_modR_fs, 
                     test_fs[,-1], 
                     type="class")
svm.test.accR_fs <- confusionMatrix(svm.testR_fs, 
                                 test_fs$RainTomorrow, 
                                 mode="prec_recall")

cbind(train=svm.train.accR_fs$overall, test=svm.test.accR_fs$overall)
cbind(train=svm.train.accR_fs$byClass, test=svm.test.accR_fs$byClass)

#-------------------------------------------
# save RData
save.image("Clustering_SVM_Workspace.RData")
