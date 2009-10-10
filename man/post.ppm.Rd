\name{post.ppm}
\alias{post.ppm}
\concept{PPM Image}
\title{Posterior Probability Map (PPM) Image}
\description{
  \code{post.ppm} 
	depicts the PPM image of voxel activations in a slice.}
\usage{post.ppm(out, slicedata, vreg = 2, swap = F, plot=T)}
\arguments{
  \item{out}{output of MCMC simulation}
  \item{slicedata}{input slicedata used in simulation.}
  \item{vreg}{regression variable to represent in PPM; default(vreg=2).}
  \item{swap}{image in radiological convention (default FALSE).}
  \item{plot}{show ppm images (with overlay) ?: (default TRUE).}}
\details{
Use the MCMC draws to estimate the Posterior Probability Map (PPM) Image.
The number of regression variables used in the simulation is equal to the number of columns specified in the design matrix, plus an intercept term; vreg=1 represents the intercept term in regression.}
\value{
  a list containing
  \item{ppm}{ppm image as matrix}
  \item{nactive}{n. of active voxels}}
\references{
Adelino R. Ferreira da Silva "A Bayesian Multilevel Model for fMRI Data Analysis", to be published.}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}}
  \code{\link{regpostsim}}
  \code{\link{post.simul.hist}}
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
post.ppm(out=out, slicedata=slicedata, vreg=2)
post.ppm(out=out, slicedata=slicedata, vreg=4)
}}
\keyword{dplot}
