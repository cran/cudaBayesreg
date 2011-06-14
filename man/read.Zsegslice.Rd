\name{read.Zsegslice}
\alias{read.Zsegslice}
\concept{Read group information (Z data) for random effects estimation based on three brain segmented regions: CSF, gray and white matter.
}
\title{ Read brain segmented data based on structural regions for CSF, gray, and white matter.
}  
\description{
  \code{read.Zsegslice} 
  builds the Z matrix of the statistical model, based on the brain segmented
	regions CSF/GRY/WHT for a given fMRI dataset.
}
\usage{read.Zsegslice(slicedata, ymaskdata)}
\arguments{
	\item{slicedata}{list of data values returned by the call to \code{read.fmrislice}}
	\item{ymaskdata}{list of data values returned by the call to \code{premask}}
}
\details{The FSL tools may be used to obtain the segmented masks for brain 
  parcellation in three main regions according to tissue type (CSF,GRY,WHT).
  If \code{fbase} has been left unspecified in reading \code{slicedata}, 
  then three user specified segmented datasets in gzipped NIFTI format must be provided as input,
  with the names \file{csf.nii.gz}, \file{gry.nii.gz}, and \file{wht.nii.gz}.
  Otherwise, \code{fbase} indicates the dataset prefix of one of the three segmented masks
  provided as a group effects example for \file{swrfM_filtered_func_data.nii.gz}:\cr
  \file{swrfM_csf.nii.gz},	\file{swrfM_gry.nii.gz}, and \file{swrfM_wht}.
  Only \samp{swrfM} segmented masks are provided in \pkg{cudaBayesregData}.
	The segmented masks included in the package were obtained by applying FAST to the
	structural high-resolution image \file{sM00223*}, followed by FLIRT for low-resolution
	registration to \file{fM00223*}.
	The \file{sM00223*} and \file{fM00223*} datasets are available from the SPM site, and are
	described in chapter 28 of the SPM8 manual.
	The fMRI dataset \file{swrfM_filtered_func_data.nii.gz} is a filtered version of \file{fM00223*}.
}
\value{
	\item{Z}{centered matrix specifying the characteristics of cross-sectional units (optional group information)}
}
\references{
FSL/FEAT Analysis tool, FMRIB Software Library (FSL) (\url{www.fmrib.ox.ac.uk/fsl})\cr

\emph{SPM8 Manual}, John Ashburner et. al., Functional Imaging Laboratory,
Institute of Neurology, UCL, London, (\url{http://www.fil.ion.ucl.ac.uk/spm/})
}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
Faculdade de Ciencias e Tecnologia,\cr \email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}},
  \code{\link{post.randeff}},
  \code{\link{premask}},
  \code{\link{post.ppm}}
}
\examples{
\dontrun{
fbase <- "swrfM"
slice <- 21
slicedata <- read.fmrislice(fbase=fbase, slice=slice, swap=FALSE )
ymaskdata <- premask(slicedata)
Z <- read.Zsegslice(slicedata, ymaskdata )
## Random effects simulation
fsave <- paste(tempdir(),"/simultest2",fileext = ".sav", sep="")
out <- cudaMultireg.slice(slicedata, ymaskdata, R=2000, keep=5, nu.e=3,
  fsave=fsave, zprior=TRUE)
post.ppm(out=out, slicedata=slicedata, ymaskdata=ymaskdata, vreg=2)

}}
\keyword{IO}
