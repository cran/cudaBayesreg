\name{post.simul.hist}
\alias{post.simul.hist}
\concept{MCMC}
\concept{plot}
\concept{hierarchical model}
\title{ Histogram of the posterior distribution of a regression coefficient }
\description{
   \code{post.simul.hist} Postprocessing of MCMC simulation. Histogram of the posterior distribution of regression coefficient beta[vreg].}
\usage{
	post.simul.hist(out,vreg = 2)}
\arguments{
  \item{out}{List of output objects of MCMC simulation}
  \item{vreg}{regression variable to map; default (vreg=2)}}
\details{Post-process analysis}
\seealso{
  \code{\link{cudaMultireg.slice}}
  \code{\link{regpostsim}}
  \code{\link{post.ppm}}
  \code{\link{post.tseries}}}
\examples{
\dontrun{
slicedata <- read.fmrisample(slice=3)
fsave <- "/tmp/simultest.sav"
answ <- readline("Run MCMC simulation first ? ")
run <- FALSE
if (substr(answ, 1, 1) == "y") { run <- TRUE }
if(run) {
	out <- cudaMultireg.slice(slicedata, fsave=fsave )
} else {
	load(file=fsave)
	cat("loaded",fsave,"\n")
}
post.simul.hist(out=out, vreg=2)
post.simul.hist(out=out, vreg=4)
}}
\keyword{ hplot }
