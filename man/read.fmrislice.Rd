\name{read.fmrislice}
\alias{read.fmrislice}
\concept{read fmri data}
\title{ Read fMRI data}
\description{
  \code{read.fmrislice} 
	reads pre-filtered fMRI data, mask data, and the design matrix to be used
	in fMRI data processing.}
\usage{read.fmrislice(fbase, slice, swap = TRUE)}
\arguments{
  \item{fbase}{Indicates the dataset prefix of the fMRI dataset to use.
	The prefix applies to data files: \file{\{fbase\}_filtered.nii.gz},
    \file{\{fbase\}_mask.nii.gz}, and \file{\{fbase\}_design.txt}.\cr
	Two test datsets are included in the package; one with prefix \samp{fmri},
		the other with prefix \samp{swrfM}.\cr
	The prefix \samp{swrfM} is used in the random effects example.}
  \item{slice}{The number of the slice to use.}
	\item{swap}{boolean \{T,F\}; default \samp{T} for RL data display convention consistent with FSLVIEW}
}
\details{The FSL/FEAT Analysis tool may be used to generate the prefiltered fMRI data
	 (niislicets), the mask (mask), and the design matrix (dsgn) required as data input.
}
\value{a list containing
	\item{fbase}{dataset prefix of the dataset used in the analysis}
	\item{slice}{slice number}
	\item{niislicets}{slice data at all timepoints}
	\item{mask}{slice mask}
	\item{X}{full design matrix}
	\item{nvar}{number of regression variables}
	\item{nobs}{number of observations}
	\item{swap}{relative orientation used in the data setup}
}
\references{
	R-package \pkg{dcemri},\cr

	FSL/FEAT Analysis tool, FMRIB Software Library (FSL) (\url{www.fmrib.ox.ac.uk/fsl})
}

\author{A. Ferreira da Silva, Universidade Nova de Lisboa,
Faculdade de Ciencias e Tecnologia,\cr \email{afs@fct.unl.pt}.
}
\seealso{
  \code{\link{cudaMultireg.slice}}
  \code{\link{premask}}}
\examples{
\dontrun{
slicedata <- read.fmrislice(slice=3)
print(str(slicedata))
}}

\keyword{IO}
