\name{cudaMultireg.slice}
\alias{cudaMultireg.slice}
\concept{bayes}
\concept{MCMC}
\concept{Gibbs Sampling}
\concept{hierarchical models}
\concept{linear model}

\title{CUDA Parallel Implementation of a Bayesian Multilevel Model for fMRI Data Analysis}
\description{
  \code{cudaMultireg.slice} provides an interface to a CUDA implementation
		of a Bayesian multilevel model for the analysis of brain fMRI data.
}
\usage{
cudaMultireg.slice(slicedata, ymaskdata, R, keep = 5, nu.e = 3,
	fsave = NA, zprior=FALSE)
}
\arguments{
  \item{slicedata}{list(slice=slice, niislicets=niislicets, mask=mask, dsgn=dsgn);
		input slice data used in simulation as returned by \code{read.fmrislice}}
  \item{ymaskdata}{list(yn = yn, kin = kin, nreg = nreg);
		masked and standardized slice data as returned by \code{premask}}
  \item{R}{number of MCMC draws}
  \item{keep}{MCMC thinning parameter: keep every keepth draw (def: 5)}
  \item{nu.e}{d.f. parameter for regression error variance prior (def: 3)}
	\item{fsave}{filename for saving the MCMC simulation (def: \samp{NULL} do not save)}
	\item{zprior}{boolean \{T,F\}; default \{F\} - use just a mean for \samp{Z}}
}
\details{
	The statistical model implemented in CUDA was specified as a Gibbs Sampler for hierarchical linear models with a normal prior.
	This model has been analysed by Rossi, Allenby and McCulloch in \emph{Bayesian Statistics and Marketing}, Chapter 3,
		and is referred to as \code{rhierLinearModel} in the R-package \pkg{bayesm}.
	The main computational work is done in parallel on a CUDA capable GPU. Each thread is responsible for fitting a general linear model at each voxel.
	The CUDA implementation has the following system requirements: nvcc NVIDIA Cuda Compiler driver, g++ GNU compiler (nvcc compatible version).
	The package includes source code files to build the library \file{newmat11.so}.
	This is a matrix library by R. B. Davies used in the package's host C++ code.}
\value{
  a list containing
  \item{betadraw}{nreg x nvar x R/keep array of individual regression coef draws}
  \item{taudraw}{R/keep x nreg array of error variance draws}
  \item{Deltadraw}{R/keep x nz x nvar array of Deltadraws}
  \item{Vbetadraw}{R/keep x nvar*nvar array of Vbeta draws}
}
\note{
	The statistical model may be specified as follows.

	Model: length(regdata) regression equations. \cr
        \eqn{y_i = X_ibeta_i + e_i}. \eqn{e_i} \eqn{\sim}{~} \eqn{N(0,tau_i)}.  nvar X vars in each equation. 

	Priors:\cr
    \eqn{tau_i} \eqn{\sim}{~} nu.e*\eqn{ssq_i/\chi^2_{nu.e}}.  \eqn{tau_i} is the variance of \eqn{e_i}.\cr
    \eqn{beta_i} \eqn{\sim}{~} N(ZDelta[i,],\eqn{V_{beta}}). \cr
    Note:  ZDelta is the matrix Z * Delta; [i,] refers to ith row of this product.

    \eqn{vec(Delta)} given \eqn{V_{beta}} \eqn{\sim}{~} \eqn{N(vec(Deltabar),V_{beta} (x) A^{-1})}.\cr
    \eqn{V_{beta}} \eqn{\sim}{~} \eqn{IW(nu,V)}. \cr
    \eqn{Delta, Deltabar} are nz x nvar.  \eqn{A} is nz x nz.  \eqn{V_{beta}} is nvar x nvar.
        
  By default we suppose that we don't have any z vars, Z=iota (nreg x 1).

	Simulated objects are specified as in \pkg{bayesm} with classes \samp{bayesm.mat} and \samp{bayesm.var}.
	S3 methods to summarize marginal distributions given an array of draws are then compatible with those of \pkg{bayesm} (see Examples).\cr
	Summaries will be invoked by a call to the generic summary function as in summary(object) where object is of class \samp{bayesm.mat} or \samp{bayesm.var}.\cr
	A new S3 method (\code{hcoef.post}) is specified for dispatching \samp{betadraw} plots.
}
\author{ Adelino Ferreira da Silva, Universidade Nova de Lisboa, Faculdade de Ciencias e Tecnologia, Portugal, \email{afs@fct.unl.pt}.
}
\references{
Adelino Ferreira da Silva, \emph{A Bayesian Multilevel Model for fMRI Data Analysis}, submitted for publication.

Rossi, Allenby and McCulloch, \emph{Bayesian Statistics and Marketing}, Chapter 3.\cr
\url{http://faculty.chicagogsb.edu/peter.rossi/research/bsm.html}

Davies, R.B. (1994) \emph{Writing a matrix package in C++}. In OON-SKI'94: The second annual object-oriented numerics conference, pp 207-213. Rogue Wave Software, Corvallis.\cr 
\url{http://www.robertnz.net/cpp_site.html}.
}
\seealso{
  \code{\link{read.fmrislice}},
  \code{\link{read.Zsegslice}},
  \code{\link{premask}},
  \code{\link{pmeans.hcoef}},
  \code{\link{regpostsim}},
  \code{\link{plot.hcoef.post}},
  \code{\link{post.simul.hist}},
  \code{\link{post.ppm}},
  \code{\link{post.tseries}},
  \code{\link{post.randeff}},
  \code{\link{post.shrinkage.mean}}
}

\examples{
\dontrun{
## Simulation using the visual/auditory test dataset "fmri"  
slicedata <- read.fmrislice(fbase="fmri", slice=3, swap=FALSE)
ymaskdata <- premask(slicedata)
fsave <- "/tmp/simultest1.sav"
out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=3,
  fsave=fsave, zprior=FALSE )
## Post-processing simulation
post.ppm(out=out, slicedata=slicedata, ymaskdata=ymaskdata, vreg=2)
post.ppm(out=out, slicedata=slicedata, ymaskdata=ymaskdata, vreg=4)
## "bayesm" summaries 
require("bayesm")
summary(out$betadraw)
summary(out$Deltadraw)
plot(out$Deltadraw)
summary(out$Vbetadraw)
##
## Random effects simulation using the SPM auditory dataset "swrfM*" 
fbase <- "swrfM"
slice <- 21
slicedata <- read.fmrislice(fbase=fbase, slice=slice, swap=FALSE )
ymaskdata <- premask(slicedata)
fsave <- "/tmp/simultest3.sav"
out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=3,
  fsave=fsave, zprior=TRUE)
post.ppm(out=out, slicedata=slicedata, ymaskdata=ymaskdata, vreg=2)
}
}
\keyword{ regression }
