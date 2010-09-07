\name{plot.bayesm.mat}
\alias{plot.bayesm.mat}
\concept{MCMC}
\concept{S3 method}
\concept{plot}
\title{Plot Method for Arrays of MCMC Draws}
\description{
   \code{plot.bayesm.mat} is an S3 method to plot sequence plots of MCMC draws and acfs.
   The columns in the array correspond to parameters and the rows to MCMC draws.
}
\usage{
\method{plot}{bayesm.mat}(x,names, ...)
}
\arguments{
  \item{x}{ An object of either S3 class, bayesm.mat, or S3 class, mcmc }
  \item{names}{optional character vector of names for coefficients}
  \item{...}{ standard graphics parameters }
}
\details{
  Typically, \code{plot.bayesm.mat} will be invoked by a call to the generic plot function as in
  \code{plot(object)} where object is of class bayesm.mat.
  This function is a simplified version of the equivalent function in \pkg{bayesm}.
  The original \pkg{bayesm::plot} function may be used instead in the \pkg{cudaBayesreg} context.
  See description of similar function in \code{bayesm::plot.bayesm.mat}.}
\author{ Peter Rossi, Graduate School of Business, University of Chicago. }
\keyword{ hplot }

