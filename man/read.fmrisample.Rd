\name{read.fmrisample}
\alias{read.fmrisample}
\concept{read fmri data}
\title{ Read fMRI data}
\description{
  \code{read.fmrisample} 
	reads pre-filtered fMRI data, mask data, and the design matrix to be used in fMRI data processing.}
\usage{read.fmrisample(slice)}
\arguments{
  \item{slice}{The number of the slice to use}}
\details{The FSL/FEAT Analysis tool may be used to generate the prefiltered fMRI data (niislicets), the mask (mask), and the design matrix (dsgn) required as data input.}
\value{list(slice,niislicets,mask,dsgn).}
\references{
R-package "AnalyzeFMRI",\cr
FSL/FEAT Analysis tool, FMRIB Software Library (FSL) (\url{www.fmrib.ox.ac.uk/fsl})}
\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
Faculdade de Ciencias e Tecnologia,\cr \email{afs@fct.unl.pt}.}
\seealso{
  \code{\link{cudaMultireg.slice}}
  \code{\link{premask}}}
\examples{
slicedata <- read.fmrisample(slice=3)
print(str(slicedata))
}
\keyword{IO}
