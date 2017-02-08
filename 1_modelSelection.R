setwd('/Users/darrenho/Box Sync/teaching/STAT6395/')
####
set.seed(1)
n = 2000
p = 500
X = matrix(rnorm(n*p),nrow=n,ncol=p)
X[,1] = 1
format(object.size(X),units='auto')#memory used by X

b = rep(0,p)
b[1:10] = 7

Xdf = data.frame(X)

Y = X %*% b + rnorm(n)

hatBeta = coef(lm(Y~X-1)) #Here, the [-1] ignores the intercept
print(hatBeta[1:15])

#Using out-of-core technique
write.table(X[1:500,],file='Xchunk1.txt',sep=',',row.names=F,col.names=names(Xdf))  
write.table(X[501:1000,],file='Xchunk2.txt',sep=',',row.names=F,col.names=names(Xdf))  
write.table(X[1001:1500,],file='Xchunk3.txt',sep=',',row.names=F,col.names=names(Xdf))  
write.table(X[1501:2000,],file='Xchunk4.txt',sep=',',row.names=F,col.names=names(Xdf))  

write.table(Y[1:500],file='Ychunk1.txt',sep=',',row.names=F,col.names=F)  
write.table(Y[501:1000],file='Ychunk2.txt',sep=',',row.names=F,col.names=F)  
write.table(Y[1001:1500],file='Ychunk3.txt',sep=',',row.names=F,col.names=F)  
write.table(Y[1501:2000],file='Ychunk4.txt',sep=',',row.names=F,col.names=F)  

###
# load/install biglm package
###
if(!require(biglm,quietly=TRUE)) {install.packages('biglm');require(biglm)}

Xchunk = read.table(file='Xchunk1.txt',sep=',',header=T)  
Ychunk = scan(file='Ychunk1.txt',sep=',')

form = as.formula(paste('Ychunk ~ -1 + ',paste(names(Xchunk),collapse=' + '),collapse=''))
out.biglm = biglm(formula = form,data=Xchunk)

head(hatBeta)
head(coef(out.biglm))

Xchunk = read.table(file='Xchunk2.txt',sep=',',header=T)  
Ychunk = scan(file='Ychunk2.txt',sep=',')
out.biglm = update(out.biglm,moredata=Xchunk)

head(hatBeta)
head(coef(out.biglm))

Xchunk = read.table(file='Xchunk3.txt',sep=',',header=T)  
Ychunk = scan(file='Ychunk3.txt',sep=',')
out.biglm = update(out.biglm,moredata=Xchunk)

head(hatBeta)
head(coef(out.biglm))

# Can you figure out the final step? Have we updated on all of the chunks?

Xchunk = read.table(file='Xchunk4.txt',sep=',',header=T)  
Ychunk = scan(file='Ychunk4.txt',sep=',')
out.biglm = update(out.biglm,moredata=Xchunk)

head(hatBeta)
head(coef(out.biglm))


### Forward Selection

#regsubsets
if(!require(leaps)){install.packages('leaps');require(leaps)}
out.for   = regsubsets(x=X,y=Y,nvmax=p,method='forward',intercept=FALSE)
sum.for   = summary(out.for)
model.for = sum.for$which[which.min(sum.for$cp),]
which(model.for)

#not using regsubsets
GICf = function(ind,gicType = 'AIC', sigmaSq = NULL,outOfCore = FALSE){
  if(outOfCore){
    grabVec = rep('NULL',p)
    grabVec[ind] = NA
    featureMat = read.csv('featureMat.csv', colClasses=grabVec)
    lm.out = lm(Y~.-1,data=featureMat)
  }else{
    lm.out = lm(Y~X[,ind]-1)  
  }
  if(gicType == 'AIC'){
    scaleTerm = 2
  }else if(gicType == 'BIC'){
    scaleTerm = log(n)
  }else{stop('Only supports AIC or BIC')}
  
  if(!is.null(sigmaSq)){
    if(class(sigmaSq) != class(1) | sigmaSq < 0){stop('Invalid variance estimate')}
    return(sum(lm.out$residuals**2)/n + scaleTerm/n * length(ind)*sigmaSq )  
  }else{
    return(n*log(sum(lm.out$residuals**2)/n) + scaleTerm * length(ind) )  
  }
}

write.csv(x=X,file='featureMat.csv')
p         = ncol(X)
n         = nrow(X)
sigmaSq   = NULL#Try sigmaSq = out.biglm$qr$ss/(n-p)
gicType   = 'AIC'
outOfCore = FALSE### To do forward selection out of core, will be slow

GIC          = Inf#initialize
indSelect    = c(1)#initialize
indSet       = 2:p#initialize
importantVar = 0#initialize
addedNewVar  = FALSE#initialize

repeat{
  cat('We have selected thus far: ',indSelect,'\n')
  countFeatures = 0
  indSetSweep = 0#this gets the index in indSet of importantVar
  for(j in indSet){
    indSetSweep = indSetSweep + 1
    countFeatures = countFeatures + 1
    if(countFeatures %% round(length(indSet)/5) == 0){
      cat('We have looked at the first: ', countFeatures/length(indSet),' fraction of features \n')  
    }
    indTmp = c(indSelect,j)
    GICnew = GICf(indTmp, gicType = gicType, 
                  sigmaSq = sigmaSq,outOfCore = outOfCore)
    if(GICnew < GIC){
      GIC = GICnew
      importantVar      = j
      importantVarIndex = indSetSweep
      addedNewVar       = TRUE
    }
    
  }
  if(!addedNewVar){
    break
  }else{
    indSet    = indSet[-importantVarIndex]
    indSelect = c(indSelect,importantVar)
  }
  print(GIC)
  addedNewVar = FALSE
}

### To do forward selection out of core
