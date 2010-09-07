#
# Builds NIFTI volume : may use data for a range of slices only;
#  the others are assumed zero
# If rg == NULL build volume with all slices
#
buildzstat.volume <-
function(fbase="swrfM", vreg=2, nu.e=3,  rg=c(NULL,NULL), swap=FALSE, blobsize=3, savedir="/tmp")
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
	volppm <- array(0, dim=d[1:3])
	nactive <- NULL ## active voxels per slice
	for (sl in (first:last)) {
		slicedata <- read.fmrislice(fbase=fbase, slice=sl, swap=swap )
		ymaskdata <- premask(slicedata)
		mcmcfile <- paste(savedir,"/",fbase,"-s",sl,"-nu",nu.e,".sav",sep="")
		out <- NULL
		load(mcmcfile)
		cat("loaded",mcmcfile,"\n")
		res <- post.ppm(out=out, slicedata=slicedata, ymaskdata=ymaskdata, vreg=vreg, plot=F)
		volppm[,,sl] <- res$ppm
		nactive <- c(nactive, res$nactive)
	}
	# ntotal <- sum(nactive)
	if(blobsize) { 
		##	blob regularization 
		post.ks0 <-
		function(img, blobsize=3)
		{
			d <- dim(img)
			p1 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1) # ks26
			p2 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1)
			mask <- array(c(p1,p2,p1), c(3,3,3))
			for(k in 2:(d[3]-1)) { 
				for(i in 2:(d[1]-1)) {
					for(j in 2:(d[2]-1)) { 
						kx <- img[(i-1):(i+1), (j-1):(j+1), (k-1):(k+1)] 
						stopifnot(dim(kx) == dim(mask))
						ks <- kx * mask
						sumks <- sum(ks)
						if(sum(ks) < blobsize)  img[i,j,k] <- 0
					}
				}
			}
			invisible(img)
		}
		volppm <- post.ks0(volppm, blobsize=blobsize)
	}
	zstatfname <- paste(savedir,"/",fbase,"-zstat",vreg,"-nu",nu.e,sep="")
 	volppm  <- as.nifti(volppm, value = img.nifti, verbose = TRUE)
  writeNIfTI(volppm, zstatfname, verbose=TRUE)
	cat("saved nifti volume",zstatfname,"\n")
	invisible(nactive)
}


