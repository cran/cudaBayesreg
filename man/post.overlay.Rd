\name{post.overlay}
\alias{post.overlay}
\concept{PPM visualisation}
\title{Rendering a Posterior Probability Map (PPM) volume}
\description{
  \code{post.overlay} overlays a statistical PPM volume of voxel
  activations on the original fMRI volume to visualise medical imaging
  data.
}
\usage{post.overlay(fbase="swrfM", vreg=2, nu.e=3, rg=c(NULL,NULL),
  view="axial", savedir="/tmp")
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
  \item{view}{choice among the three orthogonal views
	 \code{c("axial","coronal", "sagittal")} to use for the rendered image,
   (def: "axial").}
	\item{savedir}{Directory (def: \samp{"\tmp"}) where the (PPM) NIFTI
    volume built by \code{buildzstat.volume} is located.}
}
\references{
Adelino Ferreira da Silva, \emph{A Bayesian Multilevel Model for fMRI Data Analysis}, Computer Methods and Programs in Biomedicine, to be published.
}

\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.volume}},
  \code{\link{buildzstat.volume}},
}
\examples{
\dontrun{
## Simulation using the visual/auditory test dataset "fmri"  
cudaMultireg.volume(fbase="fmri", R=2000, savedir="/tmp")
buildzstat.volume(fbase="fmri", vreg=2)
post.overlay(fbase="fmri", vreg=2, view="axial")
buildzstat.volume(fbase="fmri", vreg=4)
post.overlay(fbase="fmri", vreg=4, view="axial")
}
}
\keyword{dplot}
