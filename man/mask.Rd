\name{mask}
\alias{mask}
\docType{data}
\title{Example of mask file used in processing the real audio-visual dataset "fmri.nii.gz"}
\description{
	The \emph{mask.nii} dataset defines the mask used in the fMRI analysis.
	The mask used here is an all-brain mask; it just removes non-brain regions, as the result of applying a brain extraction tool to the "fmri.nii.gz" dataset.
	However, other masks may be defined to select regions of interest (ROIs).
}
\usage{
fsl.mask <- system.file("data/mask.nii", package = "cudaBayesreg")
mask <- f.read.nifti.slice(file = fsl.mask, sl = 3, 1)
}
\format{
	The file "mask.nii.gz" is in ungzipped NIFTI format.
	The R-package "AnalizeFMRI" is required by "cudaBayesreg" to read gunzipped NIFTI files.
	This file may be obtained from "fmri.nii.gz" by applying FSL/FEAT of FSL/BET pre-processing tools.
	Similar preparation procedures for fMRI datasets are available in the SPM toolbox (\url{http://www.fil.ion.ucl.ac.uk/spm/}).
}
\references{
	S.M. Smith, M. Jenkinson, M.W. Woolrich, C.F. Beckmann, T.E.J. Behrens, H. Johansen-Berg, P.R. Bannister, M. De Luca, I. Drobnjak, D.E. Flitney, R. Niazy, J. Saunders, J. Vickers, Y. Zhang, N. De Stefano, J.M. Brady, and P.M. Matthews. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 23(S1):208-219, 2004

 "FSL & FreeSurfer Course", Oxford Centre for Functional Magnetic Resonance Imaging of the Brain (FMRIB), University of Oxford.\cr
  (\url{www.fmrib.ox.ac.uk/fsl}).
}
\keyword{datasets}
