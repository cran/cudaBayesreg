cudaMultireg.slice <-
function (slicedata, R = 3000, keep = 5, nu.e = 3, 
    savesim = T, fsave = "/tmp/simultest.sav") 
{
    # Mask out slice times series
    ymask <- premask(slicedata)
    ym <- ymask$ym
    kin <- ymask$kin
    stdf <- function(y) {
        return((y - mean(y))/sd(y))
    }
    yn <- apply(ym, 2, stdf) # use standardization 
    dsgn <- slicedata$dsgn
    nrcoefs <- ncol(dsgn)
    nobs <- nrow(ym)
    X0 <- as.matrix(dsgn) # fsl design matrix from fsl 
    X <- cbind(rep(1, nobs), X0[, 1:nrcoefs])
    nvar <- nrcoefs + 1
    nreg <- ncol(ym)
    nobs <- nrow(ym)
    #-----------------------------
    #  Mcmc regression stuff
    #  allocate space for the draws and set initial values of Vbeta and Delta
    nz <- 1
    Vbetadraw = matrix(double(floor(R/keep) * nvar * nvar), ncol = nvar * 
        nvar)
    Deltadraw = matrix(double(floor(R/keep) * nz * nvar), ncol = nz * 
        nvar)
    taudraw = matrix(double(floor(R/keep) * nreg), ncol = nreg)
    betadraw = array(double(floor(R/keep) * nreg * nvar), dim = c(nreg, 
        nvar, floor(R/keep)))
    runif(1)  # initialize RNG
    cat("\nBegin of Cuda call \n")
    outcuda <- .C("cudaMultireg", as.single(yn), as.single(X), 
        as.integer(nu.e), as.integer(nreg), as.integer(nobs), 
        as.integer(nvar), as.integer(R), as.integer(keep), Vbetadraw = as.single(Vbetadraw), 
        Deltadraw = as.single(Deltadraw), betadraw = as.single(betadraw), 
        taudraw = as.single(taudraw))
    cat("\nEND of Cuda call \n")
    Vbetadraw <- matrix(outcuda$Vbetadraw, ncol = nvar * nvar, 
        byrow = T)
    Deltadraw <- matrix(outcuda$Deltadraw, ncol = nz * nvar, 
        byrow = T)
    betadraw <- array(outcuda$betadraw, dim = c(nreg, nvar, floor(R/keep)))
    taudraw <- matrix(outcuda$taudraw, ncol = nreg, byrow = T)
    attributes(Vbetadraw)$class = c("cudabayesreg.var", "cudabayesreg.mat", 
        "mcmc")
    # attributes(Deltadraw)$class = c("cudabayesreg.mat", "mcmc")
    # attributes(betadraw)$class = c("cudabayesreg.hcoef")
    # attributes(taudraw)$class = c("cudabayesreg.mat", "mcmc")
    # attributes(taudraw)$mcpar = c(1, R, keep)
    # attributes(Deltadraw)$mcpar = c(1, R, keep)
    # attributes(Vbetadraw)$mcpar = c(1, R, keep)
    out <- list(Vbetadraw = Vbetadraw, Deltadraw = Deltadraw, 
        betadraw = betadraw, taudraw = taudraw)
    if (savesim) {
        cat("saving simulation ", fsave, "...")
        save(out, file = fsave)
        cat("\n")
    }
    invisible(out)
}
