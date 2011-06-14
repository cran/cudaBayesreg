\name{post.shrinkage.mean}
\alias{post.shrinkage.mean}
\title{Computes shrinkage of fitted estimates over regressions}
\description{
  \code{post.shrinkage.mean} 
	computes the mean fitted estimates as a function of the mean regression coefficient estimates over all regressions.
}
\usage{post.shrinkage.mean(out, X, vreg, plot=T)}
\arguments{
  \item{out}{output of MCMC simulation}
  \item{X}{regression matrix used in the simulation}
  \item{vreg}{number of the regression coefficient}
  \item{plot}{\{T,F\} output plot (default=T)}
}
\details{
	To assess the influence of the hyperparameter \eqn{nu} on the dispersion of the
	fitted estimates and regression coefficient estimates two plots are available in the
	package: one in terms of means values, the other in terms of maximum and minimum values.
	These plots help visualizing shrinkage by analyzing the influence of the hyperparameter
	\eqn{nu} on the estimates.
	Different shrinkage plots may be compared for simulations with different \eqn{nu}
	values.
}
\value{a list containing
	\item{yrecmean}{mean of fitted values}
	\item{beta}{mean of estimated coefficients over all regressions}
}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}}
}
\examples{
\dontrun{
slicedata <- read.fmrislice(fbase="fmri", slice=3, swap=FALSE)
ymaskdata <- premask(slicedata)
fsave <- paste(tempdir(),"/simultest1",fileext = ".sav", sep="")
nu1 <- 3
out1 <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=nu1,
  fsave=fsave1, zprior=FALSE, rng=1 )
fsave <- paste(tempdir(),"/simultest2",fileext = ".sav", sep="")
nu2 <- slicedata$nobs
out2 <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=nu2,
  fsave=fsave2, zprior=FALSE, rng=1 )
vreg <- 2
x1 <- post.shrinkage.mean(out1, slicedata$X, vreg=vreg, plot=F) 
x2 <- post.shrinkage.mean(out2, slicedata$X, vreg=vreg, plot=F) 
par(mfrow=c(1,2), mar=c(4,4,1,1)+0.1)
xlim=range(c(x1$beta, x2$beta))
ylim=range(c(x1$yrecmean, x2$yrecmean))
plot(x1$beta, x1$yrecmean,type="p", pch="+", col="violet", ylim=ylim,
  xlim=xlim, xlab=expression(beta), ylab="y")
legend("topright", expression(paste(nu,"=3")), bg="seashell")
plot(x2$beta, x2$yrecmean,type="p", pch="+", col="blue", ylim=ylim,
  xlim=xlim, xlab=expression(beta), ylab="y")
legend("topright", expression(paste(nu,"=45")), bg="seashell")
par(mfrow=c(1,1))
}}
\keyword{dplot}
