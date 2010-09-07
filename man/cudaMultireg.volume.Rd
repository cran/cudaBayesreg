\name{cudaMultireg.volume}
\alias{cudaMultireg.volume}
\concept{bayes}
\concept{MCMC}
\concept{Gibbs Sampling}
\concept{hierarchical models}
\concept{linear model}

\title{CUDA Parallel Implementation of a Bayesian Multilevel Model for fMRI Data Analysis on a fMRI NIFTI volume}
\description{
  \code{cudaMultireg.volume} provides an interface to a CUDA implementation
		of a Bayesian multilevel model for the analysis of brain fMRI data.
    Data is processed on a slice-by-slice basis.
    Data volumes in gzipped NIFTI format are used.
}
\usage{
cudaMultireg.volume(fbase="swrfM", R=2000, keep=5, nu.e=3,
  zprior=FALSE, rng=0, rg=c(NULL,NULL), swap=FALSE, savedir="/tmp")
}
\arguments{
  \item{fbase}{Indicates the data set prefix of the fMRI data set to use.
	The prefix applies to data files: \file{\{fbase\}_filtered.nii.gz},
    \file{\{fbase\}_mask.nii.gz}, and \file{\{fbase\}_design.txt}.\cr
	Two test data sets are included in the package; one with prefix \samp{fmri},
		the other with prefix \samp{swrfM}.\cr
	The prefix \samp{swrfM} is used in the random effects example.}
  \item{R}{number of MCMC draws}
  \item{keep}{MCMC thinning parameter: keep every keepth draw (def: 5)}
  \item{nu.e}{d.f. parameter for regression error variance prior (def: 3)}
	\item{zprior}{Boolean \{T,F\}; default \{F\} - use just a mean for \samp{Z}
   (see model description in \code{cudaMultireg.slice}. }
	\item{rng}{integer \{0,1,2\}: type of RNG to use \{0-Marsaglia Multicarry,
    1-R. P. Brent xorgens, 2-Mersenne Twister MT19937-64\}; (def. 0-Marsaglia Multicarry)}
	\item{rg}{rg=c(first, last): a vector containing the first and last numbers
    of the sequence of slices to be processed.
    If rg=c(NULL,NULL) (default), all slices in the volume are processed.}
  \item{swap}{ logical variable (default = \samp{FALSE}) for choosing the
    right/left data display convention consistent with FSLVIEW } 
	\item{savedir}{Directory (def: \samp{"\tmp"}) were the MCMC simulations for all slices are
    going to be saved.}
}
\details{
	The statistical model implemented in CUDA was specified as a Gibbs Sampler for hierarchical linear models
    with a normal prior.
	The main computational work is done in parallel on a CUDA capable GPU. Each thread is responsible for fitting
    a general linear model at each voxel.
  Data volumes are processed on a slice-by-slice basis, before reconstructing the processed volume, using \code{build.zstatvolume}.
	The statistical model is specified in \code{cudaMultireg.slice}. 
  To run the examples, the data sets from the R-package \pkg{cudaBayesregData} are required.
}
\author{ Adelino Ferreira da Silva, Universidade Nova de Lisboa, Faculdade de Ciencias e Tecnologia, Portugal, \email{afs@fct.unl.pt}.
}
\references{
Adelino Ferreira da Silva, \emph{A Bayesian Multilevel Model for fMRI Data Analysis}, Computer Methods and Programs in Biomedicine, to be published.

Rossi, Allenby and McCulloch, \emph{Bayesian Statistics and Marketing}, Chapter 3.\cr
\url{http://faculty.chicagogsb.edu/peter.rossi/research/bsm.html}

Davies, R.B. (1994) \emph{Writing a matrix package in C++}. In OON-SKI'94: The second annual object-oriented numerics conference,
pp 207-213. Rogue Wave Software, Corvallis.\cr 
\url{http://www.robertnz.net/cpp_site.html}.

Richard. P. Brent, \emph{Some long-period random number generators using shifts and xors}, Preprint: 2 July 2007.

Brandon Whitcher, Volker Schmid and Andrew Thornton, Package \bold{oro.nifti}: Rigorous - NIfTI Input / Output, 2010.
}

\seealso{
  \code{\link{cudaMultireg.slice}},
  \code{\link{buildzstat.volume}},
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
## simulation using the SPM auditory dataset "swrfM*" 
cudaMultireg.volume(fbase="swrfM", R=2000, rg=c(13,16), savedir="/tmp")
buildzstat.volume(fbase="swrfM", rg=c(13,16))
post.overlay(fbase="swrfM", vreg=2, rg=c(13,16), view="axial")
##
## Random effects simulation using the SPM auditory dataset "swrfM*" 
cudaMultireg.volume(fbase="swrfM", R=2000, zprior=TRUE, rng=1,
  rg=c(17,21),  savedir="/tmp")
buildzstat.volume(fbase="swrfM", rg=c(17,21))
post.overlay(fbase="swrfM", vreg=2, rg=c(17,21), view="axial")
##
## Simulation using the visual/auditory test dataset "fmri"  
cudaMultireg.volume(fbase="fmri", R=2000, savedir="/tmp")
buildzstat.volume(fbase="fmri", vreg=2)
post.overlay(fbase="fmri", vreg=2, view="axial")
buildzstat.volume(fbase="fmri", vreg=4)
post.overlay(fbase="fmri", vreg=4, view="axial")
}
}
\keyword{ regression }
