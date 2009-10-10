\name{post.tseries}
\alias{post.tseries}
\title{Show fitted time series of active voxel}
\description{
  \code{post.tseries} 
	plots the fitted time series of a voxel estimated as active.}
\usage{post.tseries(out, slicedata, vreg=2)}
\arguments{
  \item{out}{output of MCMC simulation}
  \item{slicedata}{input slicedata used in simulation.}
  \item{vreg}{number of the active variable to vizualize; default(vreg=2).}}
\details{
	Use the estimated regression coefficients to vizualize the fitted time series in an active voxel.}
\references{
Adelino R. Ferreira da Silva "A Bayesian Multilevel Model for fMRI Data Analysis", to be published.}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}}
  \code{\link{post.simul.hist}}
  \code{\link{post.simul.betadraw}}
  \code{\link{post.ppm}}}
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
post.tseries(out=out, slicedata=slicedata)
}}

\keyword{dplot}
