\name{cudaMultireg.slice}
\alias{cudaMultireg.slice}
\concept{bayes}
\concept{MCMC}
\concept{Gibbs Sampling}
\concept{hierarchical models}
\concept{linear model}

\title{CUDA Parallel Implementation of a Bayesian Multilevel Model for fMRI Data Analysis}
\description{
  \code{cudaMultireg.slice}
	This function provides an interface to a CUDA implementation of a Bayesian multilevel model for the analysis of brain fMRI data.
}
\usage{
cudaMultireg.slice(slicedata, R, keep = 5, nu.e = 3, savesim = T,
             fsave = "/tmp/simultest.sav")}
\arguments{
  \item{slicedata}{ list(slice=slice, niislicets=niislicets, mask=mask, dsgn=dsgn).}
  \item{R}{ number of MCMC draws.}
  \item{keep}{ MCMC thinning parameter: keep every keepth draw (def: 5).}
  \item{nu.e}{ d.f. parameter for regression error variance prior (def: 3).}
	\item{savesim}{ save MCMC simulation, (T,F) (def: TRUE).}
	\item{fsave}{ filename for output of MCMC simulation (def: "/tmp/simultest.sav").}
}
\details{
	The statistical model implemented in CUDA was specified as a Gibbs Sampler for hierarchical linear models with a normal prior (see references below).
	The main computational work is done in parallel on a CUDA capable GPU. Each thread is responsible for fitting a general linear model at each voxel.
	The CUDA implementation has the following system requirements: nvcc NVIDIA Cuda Compiler driver, g++ GNU compiler (nvcc compatible version). The package includes source code files to build the libraries "libcutil.so" and "newmat11.so". The first, is a shared version of the utility library used by the NVIDIA GPU Computing SDK software (see cudaBayesreg/src/Makefile for configuration). The second, is a matrix library by R. B. Davies used by the package's host C++ code.}
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
        
          By default we supose that we don't have any z vars, Z=iota (nreg x 1).
}
\author{ Adelino Ferreira da Silva, Universidade Nova de Lisboa, Faculdade de Ciencias e Tecnologia, Portugal, \email{afs@fct.unl.pt}.
}
\references{
Adelino Ferreira da Silva, \emph{A Bayesian Multilevel Model for fMRI Data Analysis}, submitted for publication.\cr

Rossi, Allenby and McCulloch, \emph{Bayesian Statistics and Marketing}, Chapter 3.
\url{http://faculty.chicagogsb.edu/peter.rossi/research/bsm.html}

Davies, R.B. (1994) Writing a matrix package in C++. In OON-SKI'94: The second annual object-oriented numerics conference, pp 207-213. Rogue Wave Software, Corvallis. 
\url{http://www.robertnz.net/cpp_site.html}.
}
\seealso{
  \code{\link{read.fmrisample}}
  \code{\link{premask}}
  \code{\link{pmeans.hcoef}}
  \code{\link{regpostsim}}
  \code{\link{plot.hcoef.post}}
  \code{\link{post.simul.hist}}
  \code{\link{post.simul.betadraw}}
  \code{\link{post.ppm}}
  \code{\link{post.tseries}}}
\examples{
\dontrun{
slicedata <- read.fmrisample(slice=3)
fsave <- "/tmp/simultest.sav"
out <- cudaMultireg.slice(slicedata, R=2000, keep=5, nu.e=3, fsave=fsave );
post.ppm(out=out, slicedata=slicedata, vreg=2)
post.ppm(out=out, slicedata=slicedata, vreg=4)
}}
\keyword{ regression }
