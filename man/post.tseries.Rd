\name{post.tseries}
\alias{post.tseries}
\title{Show fitted time series of active voxel}
\description{
  \code{post.tseries} 
	plots the fitted time series of a voxel estimated as active.}
\usage{post.tseries(out, slicedata, ymaskdata, vreg=2)}
\arguments{
  \item{out}{output of MCMC simulation}
  \item{slicedata}{list(slice=slice, niislicets=niislicets, mask=mask, dsgn=dsgn);
		input slicedata used in simulation as returned by \code{read.fmrislice}}
  \item{ymaskdata}{list(yn = yn, kin = kin, nreg = nreg);
		masked and standardised slice data as returned by \code{premask}}
  \item{vreg}{number of the active variable to visualize; default(vreg=2).}}
\details{
	Use the estimated regression coefficients to visualize the fitted time series
	in an active voxel.
}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}},
  \code{\link{post.simul.hist}},
  \code{\link{post.simul.betadraw}},
  \code{\link{post.ppm}}
}
\examples{
\dontrun{
## read data and load MCMC simulation 
slicedata <- read.fmrislice(fbase="fmri", slice=3, swap=TRUE)
ymaskdata <- premask(slicedata)
fsave <- paste(tempdir(),"/simultest1",fileext = ".sav", sep="")
load(fsave)
post.tseries(out=out, slicedata=slicedata, ymaskdata=ymaskdata, vreg=2)
}}
\keyword{dplot}
