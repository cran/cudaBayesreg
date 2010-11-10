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
buildzstat.volume(fbase="swrfM", vreg=2, nu.e=3, rg=c(NULL,NULL),
  swap=FALSE, blobsize=3, savedir="/tmp")
}
\arguments{
  \item{fbase}{Indicates the data set prefix of the fMRI data set to use.
  The prefix applies to data files: \file{\{fbase\}_filtered.nii.gz},
    \file{\{fbase\}_mask.nii.gz}, and \file{\{fbase\}_design.txt}.\cr
  Two test data sets are included in the package; one with prefix \samp{fmri},
    the other with prefix \samp{swrfM}.\cr
  The prefix \samp{swrfM} is used in the random effects example.}
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
  \item{savedir}{Directory (def: \samp{"\tmp"}) were the MCMC simulations for all
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
Adelino Ferreira da Silva, \emph{A Bayesian Multilevel Model for fMRI Data Analysis}, Computer Methods and Programs in Biomedicine, to be published.
}

\seealso{
  \code{\link{buildzstat.volume}},
  \code{\link{cudaMultireg.slice}},
  \code{\link{post.overlay}}
}

\examples{
\dontrun{
## Simulation using the visual/auditory test dataset "fmri"  
cudaMultireg.volume(fbase="fmri", R=2000, savedir="/tmp")
buildzstat.volume(fbase="fmri", vreg=2)
post.overlay(fbase="fmri", vreg=2, view="axial")
buildzstat.volume(fbase="fmri", vreg=4)
post.overlay(fbase="fmri", vreg=4, view="axial")
##
## simulation using the SPM auditory dataset "swrfM*" 
cudaMultireg.volume(fbase="swrfM", R=2000, rg=c(13,16), savedir="/tmp")
buildzstat.volume(fbase="swrfM", rg=c(13,16))
post.overlay(fbase="swrfM", vreg=2, rg=c(13,16), view="axial")
##

}
}
\keyword{ utilities }

