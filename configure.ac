AC_INIT([cudaBayesreg], 0.3-15)

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CXX=`${R_HOME}/bin/R CMD config CXX`

HAS_CUDA=FALSE
if test -z "${CUDA_HOME}"; then
  AC_MSG_ERROR([could not determine CUDA_HOME. Please see the README.html file for installation requirements])
else
  HAS_CUDA=TRUE
fi

AC_PROG_CXX
AC_PROG_INSTALL
AC_SUBST(R_HOME)
AC_SUBST(CXX)
AC_SUBST(CUDA_HOME)
AC_SUBST(HAS_CUDA)
AC_CONFIG_FILES([
  src/newmat11/Makefile
  src/Makefile])
AC_OUTPUT

