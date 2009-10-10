#
# fMRI data as prefiltered by FSL :  read all time series in slice  
# fMRI  input data:
# 	niislicets: slice time-series
#		mask: slice mask
# 	dsgn: design matrix
#

read.fmrisample <-
function (slice = 3) 
{
    fsl.filtered <- system.file("data/filtered_func_data.nii", 
        package = "cudaBayesreg")
    fsl.mask <- system.file("data/mask.nii", package = "cudaBayesreg")
    niislicets <- f.read.nifti.slice.at.all.timepoints(file = fsl.filtered, 
        slice)
    mask <- f.read.nifti.slice(file = fsl.mask, sl = slice, 1)
    # FSL Design matrix 
    fsl.design <- system.file("data/design.txt", package = "cudaBayesreg")
    dsgn <- read.table(fsl.design, hea = F)
    invisible(list(slice = slice, niislicets = niislicets, mask = mask, 
        dsgn = dsgn))
}
