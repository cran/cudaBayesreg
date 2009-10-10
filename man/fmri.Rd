\name{fmri}
\alias{fmri}
\docType{data}
\title{Example of real audio-visual dataset}
\description{
	The \emph{fmri.nii.gz} dataset is included in the \emph{FSL & FreeSurfer Course} available form the Oxford Centre for Functional Magnetic Resonance Imaging of the Brain (FMRIB), University of Oxford.
	The dataset has been used as a demonstration of the FEAT fMRI Analysis tool, provided by the FMRIB Software Library (FSL) (\url{www.fmrib.ox.ac.uk/fsl}).
	This dataset is from an audio-visual experiment. Auditory stimulation was applied as an alternating "boxcar" with 45s-on-45s-off and visual stimulation was applied as an alternating "boxcar" with 30s-on-30s-off. The TR is 3 seconds.
	The dataset includes just 45 timepoints and 5 slices from the original 4D data.
	The file "fmri.nii.gz" is provided as reference only, since it is not used directly in \emph{cudaBayesreg}.
	The file actually used in the package is the file "filtered_func_data.nii".
	This file may be obtained from "fmri.nii.gz" by applying pre-preprocessing tools, such as pre-filtering and motion correction.
	These pre-processing tools are readily available in the FSL/FEAT package.
	Similar preparation procedures for fMRI datasets are available in the SPM toolbox (\url{http://www.fil.ion.ucl.ac.uk/spm/}).
}
\format{
	The file "fmri.nii.gz" is in gzipped NIFTI format.
	The R-package "AnalizeFMRI" is required by "cudaBayesreg" to read gunzipped NIFTI files.
}
\references{
	S.M. Smith, M. Jenkinson, M.W. Woolrich, C.F. Beckmann, T.E.J. Behrens, H. Johansen-Berg, P.R. Bannister, M. De Luca, I. Drobnjak, D.E. Flitney, R. Niazy, J. Saunders, J. Vickers, Y. Zhang, N. De Stefano, J.M. Brady, and P.M. Matthews. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 23(S1):208-219, 2004

 "FSL & FreeSurfer Course", Oxford Centre for Functional Magnetic Resonance Imaging of the Brain (FMRIB), University of Oxford.\cr
  (\url{www.fmrib.ox.ac.uk/fsl}).
}
\keyword{datasets}
