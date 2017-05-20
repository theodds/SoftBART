#' Partial dependence plots for SoftBart
#' 
#' Modified version of the pdbart function from the BayesTree package. 
#' Run softbart at test observations constructed so that a plot can be created 
#' displaying the effect of a single variable or pair of variables.
#' 
#' @param X Training data covariates
#' @param Y training data response
#' @param xind variables to create the partial dependence plots for
#' @param levs List of levels of the covariates to evaluate at.
#' @param levquants Used if levs is not supplied; takes levs to be quantiles of associated predictors.
#' @param pl Create a plot?
#' @param plquants Quantiles for the partial dependence plot
#' @param ... Additional arguments passed to softbart or plot.
#' 
#' @return Plot methods do not return anything. pdbart and pd2bart return lists with components given below. 
#' 
#' \itemize{
#'   \item fd: A matrix (i,j) whose value is the ith draw of f_s(x_s) for the 
#'             jth value of x_s. 'fd' is for 'function draws'.
#'             For pdfbart2, fd is a single matrix, where the columns correspond 
#'             to all possible pairs of values for the pair of variables 
#'             indicated by xind. That is, all possible (x_i, x_j) where x_i is
#'             a value in the levs component corresponding to the first x and 
#'             x_j is a value in the levs component corresponding to the second
#'             one. The first x changes first.
#'  \item levs: The list of levels used, each component corresponding to a 
#'  variable. If argument levs was supplied it is unchanged. Otherwise, the 
#'  levels in levs are constructed using argument levquants.
#' }
#' 
#' The remaining componets returned in the list are the same as in the value of bart. 
pdsoftbart <- function(X, Y, xind = NULL, levs = NULL,
                       levquants = c(0.05, (1:9) / 10, 0.95),
                       pl=FALSE, plquants = c(0.05, 0.95), ...) {

  if(is.null(xind)) xind <- 1:ncol(X)

  n = nrow(X)
  nvar = length(xind)
  nlevels = rep(0,nvar)
  if(is.null(levs)) {
    levs = list()
    for(i in 1:nvar) {
      ux = unique(X[,xind[i]])
      if(length(ux) < length(levquants)) {
        levs[[i]] = sort(ux)
      } else {
        levs[[i]] = unique(quantile(X[,xind[i]], probs = levquants))
      }
    }
  }

  nlevels = unlist(lapply(levs, length))
  X_test = NULL
  for(i in 1:nvar) {
    for(v in levs[[i]]) {
      tmp = X
      tmp[,xind[i]] = v
      X_test = rbind(X_test, tmp)
    }
  }

  pdbart = softbart(X,Y,X_test, ...)

  fdr = list()
  cnt = 0
  for(j in 1:nvar) {
    fdrtemp=NULL
    for(i in 1:nlevels[j]) {
      cind = cnt + ((i-1)*n+1):(i*n)
      fdrtemp = cbind(fdrtemp, apply(pdbart$y_hat_test[,cind], 1, mean))
    }
    fdr[[j]] = fdrtemp
    cnt = cnt + n*nlevels[j]
  }

  if(is.null(colnames(X))) xlbs = paste('x', xind, sep='')
  else xlbs = colnames(X)[xind]

  retval = list(fd = fdr,levs = levs,xlbs=xlbs,
                bartcall=pdbart$call,yhat.train=pdbart$y_hat_train,
                sigma=pdbart$sigma,
                yhat.train.mean=pdbart$y_hat_train_mean,
                sigest=mean(pdbart$sigma),y=Y)

  class(retval) = 'pdbart'
  if(pl) plot(retval, plquants = plquants)

  return(retval)

}

plot.pdbart = function(
   x,
   xind = NULL,
   plquants =c(.05,.95),
   ...
)
{

  if(is.null(xind)) xind <- 1:length(x$fd)

   rgy = range(x$fd)
  cols <- c(muted("blue", 60, 80), muted("green"))
   for(i in xind) {
         tsum = apply(x$fd[[i]],2,quantile,probs=c(plquants[1],.5,plquants[2]))
         plot(range(x$levs[[i]]),rgy,type='n',xlab=x$xlbs[i],ylab='partial-dependence',...)
         lines(x$levs[[i]],tsum[2,],col=cols[1],type='b')
         lines(x$levs[[i]],tsum[1,],col=cols[2],type='b')
         lines(x$levs[[i]],tsum[3,],col=cols[2],type='b')
   }
}

pd2softbart = function (
  x.train, y.train,
  xind=1:2, levs=NULL, levquants=c(.05,(1:9)/10,.95),
  pl=TRUE, plquants=c(.05,.95), 
  ...
)
{
  n = nrow(x.train)
  nlevels = rep(0,2)
  if(is.null(levs)) {
    levs = list()
    for(i in 1:2) {
      ux = unique(x.train[,xind[i]])
      if(length(ux) <= length(levquants)) levs[[i]] = sort(ux)
      else levs[[i]] = unique(quantile(x.train[,xind[i]],probs=levquants))
    }
  } 
  nlevels = unlist(lapply(levs,length))
  xvals <- as.matrix(expand.grid(levs[[1]],levs[[2]]))
  nxvals <- nrow(xvals)
  if (ncol(x.train)==2){
    cat('special case: only 2 xs\n')
    x.test = xvals
  } else {
    x.test=NULL
    for(v in 1:nxvals) {
      temp = x.train
      temp[,xind[1]] = xvals[v,1]
      temp[,xind[2]] = xvals[v,2]
      x.test = rbind(x.test,temp)
    }
  }
  pdbrt = softbart(x.train,y.train,x.test,...)
  if (ncol(x.train)==2) {
    fdr = pdbrt$yhat.test
  } else {
    fdr = NULL 
    for(i in 1:nxvals) {
      cind =  ((i-1)*n+1):(i*n)
      fdr = cbind(fdr,(apply(pdbrt$y_hat_test[,cind],1,mean)))
    }
  }
  if(is.null(colnames(x.train))) xlbs = paste('x',xind,sep='')
  else xlbs = colnames(x.train)[xind]
  if('sigma' %in% names(pdbrt)) {
    retval = list(fd = fdr,levs = levs,xlbs=xlbs,
                  bartcall=pdbrt$call,yhat.train=pdbrt$y_hat_train,
                  sigma=pdbrt$sigma,
                  yhat.train.mean=pdbrt$y_hat_train_mean,sigest=mean(pdbrt$sigma),y=pdbrt$y)
  } else {
    retval = list(fd = fdr,levs = levs,xlbs=xlbs,
                  bartcall=pdbrt$call,yhat.train=pdbrt$y_hat_train,
                  y=pdbrt$y)
  }
  class(retval) = 'pd2bart'
  if(pl) plot(retval,plquants=plquants)
  return(retval)
}

plot.pd2bart = function(
  x,
  plquants =c(.05,.95), contour.color='white',
  justmedian=TRUE,
  ...
)
{
  pdquants = apply(x$fd,2,quantile,probs=c(plquants[1],.5,plquants[2]))
  qq <- vector('list',3)
  for (i in 1:3) 
    qq[[i]]  <- matrix(pdquants[i,],nrow=length(x$levs[[1]]))
  if(justmedian) {
    zlim = range(qq[[2]])
    vind = c(2)
  } else {
    par(mfrow=c(1,3))
    zlim = range(qq)
    vind = 1:3
  }
  for (i in vind) {
    image(x=x$levs[[1]],y=x$levs[[2]],qq[[i]],zlim=zlim,
          xlab=x$xlbs[1],ylab=x$xlbs[2],...)
    contour(x=x$levs[[1]],y=x$levs[[2]],qq[[i]],zlim=zlim,
            ,add=TRUE,method='edge',col=contour.color)
    title(main=c('Lower quantile','Median','Upper quantile')[i])
  }
}
