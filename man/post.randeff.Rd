\name{post.randeff}
\alias{post.randeff}
\concept{random effects}
\concept{postprocessing}
\title{Plots of the random effects distribution}
\description{
	Plots draws of the random effects distribution, following the specification
	of cross-sectional units (group information) in the Z matrix of the statistical model.
}
\usage{post.randeff(out, classnames=NULL, climits=TRUE)}
\arguments{
  \item{out}{output of MCMC simulation}
  \item{classnames}{default=NULL; concatenation of unit (class member) names used
	 in the Z matrix specification.
	 The argument may be a subvector of all unit names, but the \samp{classnames} given
	 in the argument must match the order used in the Z matrix specification.
	 If no class names are given (default) only the draws of the mean of the
	 random effects distribution are plotted.}
  \item{climits}{logical variable (default = \samp{TRUE}): if TRUE plots
   for the class draws use a commom \samp{ylim} parameter.}
}
\details{
	The statistical model allows for the analysis of random effects through the
	specification of the Z matrix in the prior,\cr 
		\eqn{beta_i} \eqn{\sim}{~} N(ZDelta[i,],\eqn{V_{beta}}).\cr 
	The example included in the package (\file{fbase="swrfM"}) defines a partition of the
	fMRI dataset in 3 classes, associated with 3 brain regions: CSF, gray matter
	and white matter (see examples).
}
\references{
Adelino Ferreira da Silva, \emph{A Bayesian Multilevel Model for fMRI Data Analysis}, Computer Methods and Programs in Biomedicine, to be published.
}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}},
	\code{\link{read.Zsegslice}},
  \code{\link{read.fmrislice}}
}
\examples{
\dontrun{
## Random effects simulation using the SPM auditory dataset "swrfM*"
fbase <- "swrfM"
slice <- 21
slicedata <- read.fmrislice(fbase=fbase, slice=slice)
ymaskdata <- premask(slicedata)
fsave <- "/tmp/simultest3.sav"
out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=3,
  fsave=fsave, zprior=TRUE, rng=1)
## show random effects for 3 classes
post.randeff(out, classnames=c("CSF","GRY","WHT"))
}}
\keyword{dplot}
