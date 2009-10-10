#
# Post.processing of MCMC simulation
# Example of fitted time series of most active voxel 
#
post.tseries <-
function(out, slicedata, vreg=2)
{
    # Mask out slice times series
    ymask <- premask(slicedata)
    ym <- ymask$ym; kin <- ymask$kin
    # use normalization
    # normalize <- function(y) {return ((y-min(y)) / (max(y)- min(y))) } # normalize data
    # yn <- apply(ym, 2, normalize) # normalize 
    # use standardization
    stdf <- function(y) {return ((y - mean(y) )/sd(y))} 
    yn <- apply(ym, 2, stdf) # use standardization
    #-----------
    dsgn <- slicedata$dsgn
    nrcoefs <- ncol(dsgn)
    nobs <- nrow(ym)
    X0 <- as.matrix(dsgn) # fsl design matrix from fsl
    X <- cbind(rep(1,nobs), X0[,1:nrcoefs]) #  nrcoefs: number of regression coeffs to use
    nvar <- nrcoefs + 1
    nreg <- ncol(ym)
    nobs <- nrow(ym)
    #-----------------------------
    # Postprocessing 
    pmeans = pmeans.hcoef(out$betadraw) 
    px <- regpostsim(pmeans, vreg=vreg, plot=F)
    pm2 <- pmeans[,vreg]
    spma <- px$spma
    spmn <- px$spmn
    #-----------------------------
     if(length(spma)) {
         pm2rg <- range(pm2)
         cat("range pm2:",pm2rg,"\n")
         pxa <- which(pm2 == pm2rg[2]) # most active
         # pxn <- which(pm2 == pm2rg[1]) # most non-active
         betabar.a <- pmeans[pxa,]
         # betabar.n <- pmeans[pxn,]
         yfa <- X%*%betabar.a
         # yfn <- X%*%betabar.n
         #--------------
         x11(width=7, height=3.5)
         # par(mfrow=c(2,1), mar=c(4,2,2,1)+0.1)
         # par(mfrow=c(2,1), mar=c(4,2,2,1)+0.1)
         #--------------
         ylim <- range(yn[,pxa])
         plot(yn[,pxa], ty="l", ylab="", main="Example of fitted time-series for active voxel", lty="dotted", ylim=ylim)
         points(yn[,pxa])
         lines(yfa)
         #----------
         # x11(width=7, height=3)
         # par(mfrow=c(2,1), mar=c(4,2,2,1)+0.1)
         # par(mar=c(4,2,2,1)+0.1)
         # plot(yn[,pxn], ty="l", ylab="", main="non-activated voxel", lty="dotted", ylim=ylim)
         # points(yn[,pxn])
         # lines(yfn)
     } else {
        cat("\nNo active voxels detected !\n");
    }
}
