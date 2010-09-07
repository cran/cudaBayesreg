## Run cudaMultireg.slice on a data volume

cudaMultireg.volume <-
function(fbase="swrfM", R=2000, keep=5, nu.e=3, zprior=FALSE, rng=0, rg=c(NULL,NULL), swap=FALSE, savedir="/tmp")
{
  fsl.filtered <- system.file(paste("data/", fbase, "_filtered_func_data.nii.gz", 
      sep = ""), package = "cudaBayesregData")
  img.nifti <- readNIfTI(fsl.filtered)
  img <- img.nifti@.Data
	d <- dim(img)
	if(is.null(rg)) {
		nslices <- d[3]
		first <- 1
		last <- nslices
	}
	else {
		first <- rg[1]
		last <- rg[2]
	}
	for (sl in (first:last)) {
		cat("\n*** multilevel slice n.", sl,"\n")
		slicedata <- read.fmrislice(fbase=fbase, slice=sl, swap=swap)
		ymaskdata <- premask(slicedata)
		fsave <- paste(savedir,"/",fbase,"-s",sl,"-nu",nu.e,".sav",sep="")
		out <- cudaMultireg.slice(slicedata, ymaskdata, R=R, keep=keep, nu.e=nu.e,
		  fsave=fsave, zprior=zprior, rng=rng)
	}
}

