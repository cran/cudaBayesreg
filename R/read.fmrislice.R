# fMRI data as prefiltered by FSL :  read all time series in slice  
# 	niislicets: slice time-series
#		mask: slice mask
# 	X: design matrix
#		nvar: n. of regression vars
#		nobs: n. of observations

read.fmrislice <-
function (fbase, slice, swap=FALSE) 
{
    fsl.filtered <- system.file(paste("data/", fbase, "_filtered_func_data.nii.gz", 
        sep = ""), package = "cudaBayesreg")
    fsl.mask <- system.file(paste("data/", fbase, "_mask.nii.gz", 
        sep = ""), package = "cudaBayesreg")
    img.nifti <- readNIfTI(fsl.filtered)
    mask.nifti <- readNIfTI(fsl.mask)
    img <- img.nifti@.Data
    mask <- mask.nifti@.Data
    X <- nrow(img)
    Xm <- nrow(mask)
    if (swap) { # swap=T to be represented as in fslview
        niislicets <- img[X:1, , slice, ]
        mask <- mask[Xm:1, , slice]
    }
    else {
        niislicets <- img[, , slice, ]
        mask <- mask[, , slice]
    }
    # Design matrix (FSL-like) 
    fsl.design <- system.file(paste("data/", fbase, "_design.txt", 
        sep = ""), package = "cudaBayesreg")
    dsgn <- read.table(fsl.design, hea = F)
    nobs <- nrow(dsgn)
    X0 <- as.matrix(dsgn)
    X <- cbind(rep(1, nobs), X0) # with intercept
    nvar <- ncol(X)
    invisible(list(fbase=fbase, slice = slice, niislicets = niislicets, mask = mask, 
        X = X, nvar = nvar, nobs = nobs, swap = swap))
}
