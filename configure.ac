AC_INIT([cudaBayesreg], 0.3-2)

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

NVCC_CCBIN=/tmp

NVCC_CC=${NVCC_CCBIN}/gcc

NVCC_CC_VERSION=`${NVCC_CC} -dumpversion`
if test -z "${NVCC_CC_VERSION}"; then
  AC_MSG_ERROR([gcc compiler version 3.4 is required for nvcc compilation in cudaBayesreg. Please see the README.html file for installation requirements])
fi

case ${NVCC_CC_VERSION} in
  4.4*)
AC_MSG_ERROR([a compatible gcc/g++ compiler version is required for nvcc compilation in cudaBayesreg. Please see the README.html file for installation requirements])
;;
esac

AC_PROG_CXX
AC_PROG_INSTALL
AC_SUBST(R_HOME)
AC_SUBST(CXX)
AC_SUBST(CUDA_HOME)
AC_SUBST(HAS_CUDA)
AC_SUBST(NVCC_CCBIN)
AC_CONFIG_FILES([
  src/newmat11/Makefile
  src/cutil/Makefile
  src/Makefile])
AC_OUTPUT

