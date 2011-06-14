\name{post.simul.hist}
\alias{post.simul.hist}
\concept{MCMC}
\concept{plot}
\concept{hierarchical model}
\title{ Histogram of the posterior distribution of a regression coefficient }
\description{
   Postprocessing of MCMC simulation. Histogram of the posterior distribution of regression coefficient beta[vreg].}
\usage{
	post.simul.hist(out,vreg = 2)
}
\arguments{
  \item{out}{list of output objects of MCMC simulation}
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
## load MCMC simulation 
fsave <- paste(tempdir(),"/simultest1",fileext = ".sav", sep="")
load(fsave)
post.simul.hist(out=out, vreg=2)
post.simul.hist(out=out, vreg=4)
}}
\keyword{ hplot }
