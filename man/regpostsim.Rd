\name{regpostsim}
\alias{regpostsim}
\concept{MCMC}
\concept{plot}
\concept{hierarchical model}
\title{Estimation of voxel activations}
\description{
   \code{regpostsim} estimates voxel activation and plots the posterior distribution of a regression coefficient.}
\usage{
regpostsim(pmeans, vreg, plot=T)}
\arguments{
  \item{pmeans}{Posterior Means of Coefficients as processed by pmeans.hcoef()}
  \item{vreg}{regression variable to process}
  \item{plot}{plot the histogram, (default=T)}
}
\details{Estimate the active and non-active voxels based on the highest posterior density (HPD) of the coefficients simulated by the multilevel method. Plot the histogram of the posterior distribution of regression coefficient "vreg"}
\seealso{
  \code{\link{cudaMultireg.slice}},
  \code{\link{pmeans.hcoef}},
  \code{\link{plot.hcoef.post}},
  \code{\link{post.simul.hist}},
  \code{\link{post.simul.betadraw}},
  \code{\link{post.ppm}},
  \code{\link{post.tseries}}
}
\examples{
\dontrun{
slicedata <- read.fmrislice(fbase="fmri", slice=3, swap=TRUE)
ymaskdata <- premask(slicedata)
fsave <- "/tmp/simultest1.sav"
answ <- readline("Run MCMC simulation first ? ")
run <- FALSE
if (substr(answ, 1, 1) == "y") { run <- TRUE }
if(run) {
	out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=3, fsave=fsave,
   	zprior=FALSE)
} else {
	load(file=fsave)
	cat("loaded",fsave,"\n")
}
##
pmeans <- pmeans.hcoef(out$betadraw) 
px <- regpostsim(pmeans, vreg=2)
pm2 <- pmeans[,vreg]
spma <- px$spma # active voxels
spmn <- px$spmn # non-active voxels
}}
\keyword{ utilities }
