\name{buildzstat.volume}
\alias{buildzstat.volume}
\concept{bayes}
\concept{linear model}
\concept{NIFTI}

\title{Build a Posterior Probability Map (PPM) NIFTI volume}
\description{
  \code{buildzstat.volume} builds a PPM statistical volume
  from slices processed by \code{cudaMultireg.volume}.
}
\usage{
buildzstat.volume(fbase=NULL, vreg=2, nu.e=3,
  rg=c(NULL,NULL), swap=FALSE, blobsize=3, savedir=tempdir())
}
\arguments{
\item{fbase}{If \code{fbase} is left unspecified (default \code{NULL}),
  then user datasets need to be provided as input. Otherwise, \code{fbase}
  indicates the dataset prefix of one of the two demo fMRI datasets to use.
  see \code{read.fmrislice} for a detailed description.}
\item{vreg}{regression variable to represent in PPM; default(vreg=2)}
\item{nu.e}{d.f. parameter for regression error variance prior (def: 3)}
\item{rg}{rg=c(first, last): a vector containing the first and last
   numbers of the sequence of slices to be processed.
   If rg=c(NULL,NULL) (default), all slices in the volume are processed.}
\item{swap}{ logical variable (default = \samp{FALSE}) for choosing the
    right/left data display convention consistent with FSLVIEW } 
\item{blobsize}{numeric value (default=\samp{3}). Applies spatial contextual
    information in a 3D immediate neighbourhood for eliminating 3D blobs
    with less than \samp{blobsize} active voxels.}
\item{savedir}{Directory (def: \samp{tempdir()}) were the MCMC simulations for all
    slices are saved.}
}
\details{
  The PPM volume is build by \code{buildzstat.volume} after all slices in 
  the desired range \samp{rg} have been processed by \code{cudaBayesreg.volume}. 
  The PPM volume has the dimension of the original volume.
  However, when a non-null range \samp{rg} is specified only the slices in
  the range are statistically processed. The remaining slices are assumed
  to contain non-activated voxels.
  To run the examples, the data sets from the R-package \pkg{cudaBayesregData} are required.
}
\value{
  \item{nactive}{numeric vector containing the number of estimated active
  voxels for each slice in range \samp{rg}.}
}
\author{ Adelino Ferreira da Silva, Universidade Nova de Lisboa, Faculdade de Ciencias e Tecnologia, Portugal, \email{afs@fct.unl.pt}.
}
\references{
Adelino R. Ferreira da Silva (2011). ``\pkg{cudaBayesreg}: Parallel Implementation of a Bayesian Multilevel Model for fMRI Data Analysis.''
  \emph{Journal of Statistical Software}, \bold{44}(4), 1--24.
  URL \url{http://www.jstatsoft.org/v44/i04/}.

Adelino Ferreira da Silva (2011). \pkg{cudaBayesregData}: \emph{Data sets for the examples used in the package \pkg{cudaBayesreg}}, R package version	0.3-10.
  URL \url{http://CRAN.R-project.org/package=cudaBayesregData}.

Adelino Ferreira da Silva (2011). ``A Bayesian Multilevel Model for fMRI Data Analysis.'', \emph{Computer Methods and Programs in Biomedicine}, \bold{102},(3), 238--252.

Adelino Ferreira da Silva (2010). ``\pkg{cudaBayesreg}: Bayesian Computation in CUDA.'', \emph{The R Journal}, \bold{2/2}, 48-55. 
  URL \url{http://journal.r-project.org/archive/2010-2/RJournal_2010-2_Ferreira~da~Silva.pdf}.

Brandon Whitcher, Volker Schmid and Andrew Thornton (2011). \pkg{oro.nifti}: \emph{Rigorous - NIfTI Input / Output}, R package version 0.2.5.
  URL \url{http://CRAN.R-project.org/package=oro.nifti}.


}

\seealso{
  \code{\link{read.fmrislice}},
  \code{\link{buildzstat.volume}},
  \code{\link{cudaMultireg.slice}},
  \code{\link{post.overlay}}
}

\examples{
\dontrun{
## Simulation using the visual/auditory test dataset "fmri"  
cudaMultireg.volume(fbase="fmri", R=2000, savedir=tempdir())
buildzstat.volume(fbase="fmri", vreg=2)
post.overlay(fbase="fmri", vreg=2, view="axial")
buildzstat.volume(fbase="fmri", vreg=4)
post.overlay(fbase="fmri", vreg=4, view="axial")
##
## simulation using the SPM auditory dataset "swrfM*" 
cudaMultireg.volume(fbase="swrfM", R=2000, rg=c(13,16), savedir=tempdir())
buildzstat.volume(fbase="swrfM", rg=c(13,16))
post.overlay(fbase="swrfM", vreg=2, rg=c(13,16), view="axial")
##
}
}
\keyword{ utilities }

