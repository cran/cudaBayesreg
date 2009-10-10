\name{premask}
\alias{premask}
\concept{Mask fmri data}
\title{Mask out voxels with constant time-series}
\description{
  \code{premask} 
	applies a pre-defined mask to a fMRI slice in order to select regions of interest (ROIs) for processing
}
\usage{premask(slicedata)}
\arguments{
  \item{slicedata}{list(slicedata).}}
\details{
Mask out fMRI nifti data as read by cudaBayesreg::read.fmrisample(slice).
Pixels with constant time series are masked out.}
\value{
  a list containing
  \item{ym}{pixel values to retain}
  \item{kin}{indices of voxels in mask}}
\references{
package="AnalyzeFMRI",\cr
FSL/FEAT Analysis tool, FMRIB Software Library (FSL) (\url{www.fmrib.ox.ac.uk/fsl})}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
	Faculdade de Ciencias e Tecnologia,\cr
	\email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}}
  \code{\link{read.fmrisample}}}
\examples{
\dontrun{
slicedata <- read.fmrisample(slice=3)
ymask <- premask(slicedata)
print(str(ymask))
}
}
\keyword{utilities}
