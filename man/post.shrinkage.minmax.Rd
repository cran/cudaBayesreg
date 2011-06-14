\name{post.shrinkage.minmax}
\alias{post.shrinkage.minmax}
\title{Computes shrinkage of fitted estimates over regressions}
\description{
  \code{post.shrinkage.minmax} 
	computes the maximum and minimum fitted estimates, as a function of the mean regression coefficient estimates over all regressions.
}
\usage{post.shrinkage.minmax(out, X, vreg, plot=T)}
\arguments{
  \item{out}{output of MCMC simulation}
  \item{X}{regression matrix used in the simulation}
  \item{vreg}{number of the regression coefficient}
  \item{plot}{\{T,F\} output plot (default=T)}

}
\details{
	The plot helps visualizing shrinkage by analyzing the influence of the hyperparameter
	\eqn{nu} on the dispersion of the fitted maximum and minimum estimates.
	Different shrinkage plots may be compared for simulations with different \eqn{nu}
	values.
}
\value{a list containing
	\item{yrecmin}{minimum values of fitted values}
	\item{yrecmax}{maximum values of fitted values}
	\item{beta}{mean of estimated coefficients over all regressions}
}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}},
  \code{\link{read.fmrislice}}
}
\examples{
\dontrun{
slicedata <- read.fmrislice(fbase="fmri", slice=3, swap=FALSE)
ymaskdata <- premask(slicedata)
fsave <- paste(tempdir(),"/simultest1",fileext = ".sav", sep="")
nu1 <- 3
out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=nu1,
  fsave=fsave1, zprior=FALSE, rng=1)
vreg <- 2
post.shrinkage.minmax(out, slicedata$X, vreg=vreg) 
}}
\keyword{dplot}
