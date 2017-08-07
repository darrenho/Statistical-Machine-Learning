prostate = read.table('../data/prostate.data',header=T)
# Variables are:
# lcavol:  log cancer volume
# lweight: log prostate weight
# age:     patient age 
# lbph:    log of amount of benign prostate hyperplasia
# svi:     seminal vesicle invasion (0,1 valued)
# lcp:     log of capsular penetration
# gleason: Gleason score
# pgg45:   Percent of Gleason scores 4 or 5
# lpsa:    log prostate specific antigen (response)

Y = prostate$lpsa
X = prostate[,names(prostate)!=c('lpsa','train')]
n = length(Y)
p = ncol(X)

library(leaps)

leaps.plot =regsubsets(Y~.,data=X, nbest=10)
pdf('../lectures/figures/leapsProstateExampleBIC.pdf')
  plot(leaps.plot,scale='bic')
dev.off()

#### Stepwise Methods
null = lm(Y~1,data=X)
full = lm(Y~.,data=X)
#Forward Stepwise
out  = step(null,scope=list(lower=null,upper=full),direction='forward')
#Backwards Stepwise
out  = step(full,direction='backward')
#Stepwise
out  = step(null,scope=list(upper=full),direction='both')

library(leaps)
regfit.for = regsubsets ( x = X,y = Y, nvmax =19,
                      method ="forward")
regfit.for.sum = summary(regfit.for)
regfit.for.sum$which[which.min(regfit.for.sum$cp),]

regfit.bac = regsubsets ( x = X,y = Y, nvmax =19,
                            method ="backward")
regfit.bac.sum = summary(regfit.bac)
regfit.bac.sum$which[which.min(regfit.bac.sum$cp),]

