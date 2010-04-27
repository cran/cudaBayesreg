\name{mask}
\alias{mask}
\docType{data}
\title{Example of mask file used in processing the visual-auditory test dataset}
\description{
	The \file{fmri_mask.nii.gz} dataset defines the mask used in the fMRI analysis for \file{fmri_filtered_data.nii.gz}.
	The mask used here is an all-brain mask; it just removes non-brain regions, as the result of applying a brain extraction tool to the \file{fmri.nii.gz} dataset.
	However, other masks may be defined to select regions of interest (ROIs).
}
\format{
	The file \file{fmri_mask.nii.gz} is in ungzipped NIFTI format.
	The R-package \pkg{oro.nifti} is required by \pkg{cudaBayesreg} to read gunzipped NIFTI files.
	This file may be obtained from \file{fmri.nii.gz} by applying FSL/FEAT of FSL/BET pre-processing tools.
	Similar preparation procedures for fMRI datasets are available in the SPM toolbox (\url{http://www.fil.ion.ucl.ac.uk/spm/}).
}
\references{
Brandon Whitcher, Volker Schmid and Andrew Thornton, Package \bold{oro.nifti}:
 Rigorous - NIfTI Input / Output, 2010.

S.M. Smith et.al., Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 23(S1):208-219, 2004.

\emph{FSL & FreeSurfer Course}, Oxford Centre for Functional Magnetic Resonance Imaging of the Brain (FMRIB), University of Oxford.
  (\url{www.fmrib.ox.ac.uk/fsl}).
}
\keyword{datasets}
