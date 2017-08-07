######################################
#Code chunk 1: load packages and data
######################################
install.packages("darch")
install.packages("e1071")
install.packages("xtable")
library(xtable)
library(darch)
library(e1071)
library(MASS)
library(fields)
library(glmnet)
setwd("~/Downloads/")
readMNIST("~/Downloads/")
load("train.Rdata")
load("test.Rdata")

######################################
#Code chunk 2: check out some plots
######################################
dim(trainData)

par(mfrow = c(3,3))
x = seq(1, 28, by = 1)
y = x
for(i in 50:58)
{
	m1 = trainData[i,]
	m1 = matrix(m1, ncol = 28, nrow = 28, byrow = T)
	#Need to flip matrix
	count = 28
	m2 = matrix(data = NA, nrow = 28, ncol = 28)
	for(j in 1:28)
	{
		m2[,count] = m1[j,]
		count = count-1
	}
	image.plot(x, y, m2)
}

######################################
# Code chunk 3: gather data for
# problem #1-b-i
######################################

#Gather the training data: 3's vs 5's
threes = trainLabels[,4]
fives = trainLabels[,6]
#how many of each?
sum(threes)
sum(fives)
#how many total?
sum(c(threes, fives))
#proportions of each
p3 = sum(threes) / sum(c(threes, fives))
p5 = sum(fives) / sum(c(threes, fives))
p3
p5
#where are they located?
index3 = which(threes == 1)
index5 = which(fives == 1)
#Gather training X's
x3 = trainData[index3,]
x5 = trainData[index5,]
Xtrain = rbind(x3,x5)
dim(Xtrain)
#Gather training Y's: code "3" as 0 and "5" as 1
Ytrain = c( rep(0, sum(threes) ), rep(1, sum(fives)) )
Ytrain = factor(Ytrain)

#Do the same for test data
load("test.Rdata")
threes = testLabels[,4]
fives = testLabels[,6]
#how many of each?
sum(threes)
sum(fives)
#how many total?
sum(c(threes, fives))
#proportions of each
p3 = sum(threes) / sum(c(threes, fives))
p5 = sum(fives) / sum(c(threes, fives))
p3
p5
#where are they located?
index3 = which(threes == 1)
index5 = which(fives == 1)
#Gather training X's
x3 = testData[index3,]
x5 = testData[index5,]
Xtest = rbind(x3,x5)
dim(Xtest)
#Gather training Y's: code "3" as 0 and "5" as 1
Ytest = c( rep(0, sum(threes) ), rep(1, sum(fives)) )

######################################
# Code chunk 4: sparse logistic reg.,
# LDA, and kernel SVM for problem #1-b-i
######################################

#Sparse logistic regression
cv1 = cv.glmnet(Xtrain, Ytrain, family = "binomial")
minl = cv1$lambda.min
m1 = glmnet(Xtrain, Ytrain, family = c("binomial"))
par(mfrow = c(2,2))
plot(m1, xvar = c("lambda"))
abline(h = 0, lty = 2)
plot(m1, xvar = c("norm"))
abline(h = 0, lty = 2)
plot(m1, xvar = c("dev"))
abline(h = 0, lty = 2)
#make predictions with the model at min. lambda
pred.m1 = predict(m1, Xtest, s = minl, type = "response")
pred.m1 = round(pred.m1)
nwrong1 = length(which( pred.m1 != Ytest))
SpLR.misclass = nwrong1/length(Ytest)

#LDA
colsd = apply(Xtrain, 2, sd)
bad = which(colsd == 0)
Xtrain1 = Xtrain[,-bad]

m2 = lda(Xtrain1, Ytrain)
plot(m2)
pred.m2 = predict(m2, Xtest[,-bad])
nwrong2 = length(which( pred.m2$class != Ytest))
LDA.misclass = nwrong2/length(Ytest)

#Kernel SVM
n = length(Ytrain)
n
#Use a subset of training data to do CV for tuning params
set.seed(2)
sub = sample(1:n, n/50)
gseq = seq(0.0001, 0.01, by = .0008)
cseq = seq(0.01, 1, by = .09)
tune.m3 = tune(svm, Xtrain[sub,], Ytrain[sub], 
ranges = list(gamma = gseq, cost = cseq))
tune.m3$best.param
summary(tune.m3.poly)

#Polynomial kernel
m3.pol = svm(Xtrain, y = Ytrain, kernel = "polynomial",
cost = 1, gamma = 0.009)
pred.m3.pol = predict(m3.pol, Xtest)
nwrong3.pol = length(which( pred.m3.pol != Ytest))
SVM.pol.misclass = nwrong3.pol/ length(Ytest)
#Radial basis kernel
m3.rad = svm(Xtrain, y = Ytrain, kernel = "radial",
cost = 1, gamma = 0.009)
pred.m3.rad = predict(m3.rad, Xtest)
nwrong3.rad = length(which( pred.m3.rad != Ytest))
SVM.rad.misclass = nwrong3.rad/ length(Ytest)
#Sigmoid kernel
m3.sig = svm(Xtrain, y = Ytrain, kernel = "radial",
cost = 1, gamma = 0.009)
pred.m3.sig = predict(m3.sig, Xtest)
nwrong3.sig = length(which( pred.m3.sig != Ytest))
SVM.sig.misclass = nwrong3.sig/ length(Ytest)

#Compare the methods
err.rate = cbind(SpLR.misclass,LDA.misclass,SVM.pol.misclass,
SVM.rad.misclass,SVM.sig.misclass)
colnames(err.rate) = c("SpLR", "LDA", "SVM-pol", "SVM-rad","SVM-sig")
err.rate
min(err.rate)
xtable(err.rate, digits = 5)

######################################
# Code chunk 5: gather data for
# problem #1-b-ii
######################################

#Gather the training data: 4's vs 9's
fours = trainLabels[,5]
nines = trainLabels[,10]
#how many of each?
#sum(four)
#sum(nines)
#how many total?
#sum(c(fours, nines))
#proportions of each
p4 = sum(fours) / sum(c(fours, nines))
p9 = sum(nines) / sum(c(fours, nines))
p4
p9
#where are they located?
index4 = which(fours == 1)
index9 = which(nines == 1)
#Gather training X's
x4 = trainData[index4,]
x9 = trainData[index9,]
Xtrain = rbind(x4,x9)
#dim(Xtrain)
#Gather training Y's: code "4" as 0 and "9" as 1
Ytrain = c( rep(0, sum(fours) ), rep(1, sum(nines)) )
Ytrain = factor(Ytrain)

#Do the same for test data
fours = testLabels[,5]
nines = testLabels[,10]
#how many of each?
#sum(fours)
#sum(nines)
#how many total?
#sum(c(fours, nines))
#proportions of each
#p4 = sum(fours) / sum(c(fours, nines))
#p9 = sum(nines) / sum(c(fours, nines))
p4
p9
#where are they located?
index4 = which(fours == 1)
index9 = which(nines == 1)
#Gather training X's
x4 = testData[index4,]
x9 = testData[index9,]
Xtest = rbind(x4,x9)
dim(Xtest)
#Gather training Y's: code "4" as 0 and "9" as 1
Ytest = c( rep(0, sum(fours) ), rep(1, sum(nines)) )

######################################
# Code chunk 6: sparse logistic reg.,
# LDA, and kernel SVM for problem #1-b-ii
######################################

#Sparse logistic regression
cv1 = cv.glmnet(Xtrain, Ytrain, family = "binomial")
minl = cv1$lambda.min
m1 = glmnet(Xtrain, Ytrain, family = c("binomial"))
par(mfrow = c(2,2))
plot(m1, xvar = c("lambda"))
abline(h = 0, lty = 2)
plot(m1, xvar = c("norm"))
abline(h = 0, lty = 2)
plot(m1, xvar = c("dev"))
abline(h = 0, lty = 2)
#make predictions with the model at min. lambda
pred.m1 = predict(m1, Xtest, s = minl, type = "response")
pred.m1 = round(pred.m1)
nwrong1 = length(which( pred.m1 != Ytest))
SpLR.misclass = nwrong1/length(Ytest)

#LDA
colsd = apply(Xtrain, 2, sd)
bad = which(colsd == 0)
Xtrain1 = Xtrain[,-bad]

m2 = lda(Xtrain1, Ytrain)
plot(m2)
pred.m2 = predict(m2, Xtest[,-bad])
nwrong2 = length(which( pred.m2$class != Ytest))
LDA.misclass = nwrong2/length(Ytest)

#Kernel SVM
n = length(Ytrain)
n
#Use a subset of training data to do CV for tuning params
set.seed(2)
sub = sample(1:n, n/50)
gseq = seq(0.0001, 0.01, by = .0008)
cseq = seq(0.01, 1, by = .09)
tune.m3 = tune(svm, Xtrain[sub,], Ytrain[sub], 
ranges = list(gamma = gseq, cost = cseq))
tune.m3$best.param
#summary(tune.m3)

#Polynomial kernel
m3.pol = svm(Xtrain, y = Ytrain, kernel = "polynomial",
cost = 0.90, gamma = 0.009)
pred.m3.pol = predict(m3.pol, Xtest)
nwrong3.pol = length(which( pred.m3.pol != Ytest))
SVM.pol.misclass = nwrong3.pol/ length(Ytest)
#Radial basis kernel
m3.rad = svm(Xtrain, y = Ytrain, kernel = "radial",
cost = 0.90, gamma = 0.009)
pred.m3.rad = predict(m3.rad, Xtest)
nwrong3.rad = length(which( pred.m3.rad != Ytest))
SVM.rad.misclass = nwrong3.rad/ length(Ytest)
#Sigmoid kernel
m3.sig = svm(Xtrain, y = Ytrain, kernel = "radial",
cost = 0.90, gamma = 0.009)
pred.m3.sig = predict(m3.sig, Xtest)
nwrong3.sig = length(which( pred.m3.sig != Ytest))
SVM.sig.misclass = nwrong3.sig/ length(Ytest)

#Compare the methods
err.rate = cbind(SpLR.misclass,LDA.misclass,SVM.pol.misclass,
SVM.rad.misclass,SVM.sig.misclass)
colnames(err.rate) = c("SpLR", "LDA", "SVM-pol", "SVM-rad","SVM-sig")
err.rate
min(err.rate)
xtable(err.rate, digits = 5)


