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
Adelino R. Ferreira da Silva (2011). ``\pkg{cudaBayesreg}: Parallel Implementation of a Bayesian Multilevel Model for fMRI Data Analysis.''
  \emph{Journal of Statistical Software}, \bold{44}(4), 1--24.
  URL \url{http://www.jstatsoft.org/v44/i04/}.

Adelino Ferreira da Silva (2011). ``A Bayesian Multilevel Model for fMRI Data Analysis.'', \emph{Computer Methods and Programs in Biomedicine}, \bold{102},(3), 238--252.

Adelino Ferreira da Silva (2010). ``\pkg{cudaBayesreg}: Bayesian Computation in CUDA.'', \emph{The R Journal}, \bold{2/2}, 48-55. 
  URL \url{http://journal.r-project.org/archive/2010-2/RJournal_2010-2_Ferreira~da~Silva.pdf}.
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
fsave <- paste(tempdir(),"/simultest3",fileext = ".sav", sep="")
out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=3,
  fsave=fsave, zprior=TRUE, rng=1)
## show random effects for 3 classes
post.randeff(out, classnames=c("CSF","GRY","WHT"))
}}
\keyword{dplot}
