\name{post.simul.betadraw}
\alias{post.simul.betadraw}
\concept{MCMC}
\concept{plot}
\concept{hierarchical model}
\title{Postprocessing of MCMC simulation}
\description{
	Postprocessing of MCMC simulation.  Boxplots of posterior distributions for regressor coefficient beta[vreg] in two cases: estimates for 30 time series of random voxels in active cortex areas; estimates for 30 time series of random voxels in non-active cortex areas.}
\usage{
	post.simul.betadraw(out,vreg = 2)}
\arguments{
  \item{out}{List of output objects of MCMC simulation}
  \item{vreg}{regression variable to map; default \samp{vreg=2}}
}
\details{Post-process analysis}
\seealso{
  \code{\link{cudaMultireg.slice}},
  \code{\link{regpostsim}},
  \code{\link{post.ppm}},
  \code{\link{post.tseries}}
}
\examples{
\dontrun{
slicedata <- read.fmrislice(fbase="fmri", slice=3, swap=FALSE)
ymaskdata <- premask(slicedata)
fsave <- paste(tempdir(),"/simultest1",fileext = ".sav", sep="")
answ <- readline("Run MCMC simulation first ? ")
run <- FALSE
if (substr(answ, 1, 1) == "y") { run <- TRUE }
if(run) {
	out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=3,
    fsave=fsave, zprior=FALSE)
} else {
	load(file=fsave)
	cat("loaded",fsave,"\n")
}
post.simul.betadraw(out=out, vreg=2)
post.simul.betadraw(out=out, vreg=4)
}}
\keyword{ hplot }
