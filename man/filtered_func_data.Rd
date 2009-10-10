\name{filtered_func_data}
\alias{filtered_func_data}
\docType{data}
\title{Example pre-processed real audio-visual dataset}
\description{
	The dataset \emph{fmri.nii} (\link{fmri}) is from an audio-visual experiment.
	The \emph{filtered_func_data.nii} is a dataset obtained from \emph{fmri.nii} by applying the usual fMRI pre-processing steps, e.g., brain extraction, motion correction, high-pass filtering, spatial smoothing, intensity normalization, etc.
	All these pre-processing steps may be applied using some external fMRI software package. The file included in the package was prepared with the FSL/FEAT software tool.
	Similar preparation procedures for fMRI datasets are available in the SPM toolbox (\url{http://www.fil.ion.ucl.ac.uk/spm/}).
}
\usage{
fsl.filtered <- system.file("data/filtered_func_data.nii",
                            package = "cudaBayesreg")
niislicets <- f.read.nifti.slice.at.all.timepoints(
                            file = fsl.filtered, slice = 3)
}
\format{
	The file "filtered_func_data.nii" is in ungzipped NIFTI format.
	The R-package "AnalizeFMRI" is required by "cudaBayesreg" to read gunzipped NIFTI files.
}
\references{
	S.M. Smith, M. Jenkinson, M.W. Woolrich, C.F. Beckmann, T.E.J. Behrens, H. Johansen-Berg, P.R. Bannister, M. De Luca, I. Drobnjak, D.E. Flitney, R. Niazy, J. Saunders, J. Vickers, Y. Zhang, N. De Stefano, J.M. Brady, and P.M. Matthews. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 23(S1):208-219, 2004

 "FSL & FreeSurfer Course", Oxford Centre for Functional Magnetic Resonance Imaging of the Brain (FMRIB), University of Oxford.\cr
  (\url{www.fmrib.ox.ac.uk/fsl}).
}
\keyword{datasets}
