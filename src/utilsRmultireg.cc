//$Id: utilsRmultireg.cc,v 1.2 2010/05/15 16:26:06 afs Exp afs $
//
#define MATHLIB_STANDALONE 1

#include <Rmath.h>

#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
                                     // newmatap.h will get include.h
#include "newmatap.h"                // need matrix applications

#include "newmatio.h"                // need matrix output routines


#ifdef use_namespace
using namespace NEWMAT;              // access NEWMAT namespace
#endif

#include "utilsFuncs.h"    

//
//===================================================================

void varnr(const Real data[], const int n, Real *var)
// Given an array of data[1..n], this routine returns its mean ave, variance var
{
	int j;
	Real ep=0.0,s;
	// printf("n must be at least 2 in moment\n"); exit(1);
	if (n <= 1) {
		cout << "n must be at least 2 in moment" << endl;
	 	// exit(1);
                return;
 	}
	s=0.0;
	(*var)=0.0;
	for (j=0;j<n;j++) {
		s=data[j];
		ep += s;
		*var += s*s;
	}
	*var=(*var-ep*ep/n)/(n-1); // Corrected two-pass formula.
}
		

//=============================================================
//

void rwishart(const int nu, const SymmetricMatrix V, Matrix* CI, SymmetricMatrix* IW)
{
	int k, a, b, df;
	int m = V.nrows();
	Matrix T(m,m);
	if(m > 1) {
		Real rchi[m];
		a=nu+nu-m+1;
		b=nu-m+1;
		for(int i=0; i<m; i++) {
			df=a-(b+i);
			rchi[i] = sqrt( rchisq(df) );
		}

		k = m*(m+1)/2-m;
		Real tmp[k];
		for (int i = 0; i < k; ++i) {
			tmp[i] = rnorm(0.,1.);
		}
		T = lowerTri(m, &tmp[0]);
		diagSub(&T, rchi);
	}
	else { df = nu; T = sqrt( rchisq(df) ); }
	Matrix U = Cholesky(V).t();
	Matrix C = crossprod(T, U);
	*CI = backsolveU(C);
	*IW << tcrossprod2(*CI,*CI);
}

//----------------------------------------------------------------------

void rmultireg(const Matrix& Y, const Matrix& X, const Matrix& Bbar, const SymmetricMatrix& A, const int nu, const Matrix& V, Matrix* B, SymmetricMatrix* IW)
{
	int n = Y.nrows();
	int m = Y.ncols();
	int k = X.ncols();
	Matrix RA = Cholesky(A);
	Matrix W = X & RA;
	Matrix tmp = RA.t()*Bbar;
	Matrix Z = Y & tmp;
	Matrix IR = backsolve(W);
	Matrix Btilde =	tcrossprod(IR, IR) * crossprod(W,Z);
	//---------------
	Matrix S, S0;
 	S0 = Z - (W * Btilde);
	S = crossprod(S0,S0);
	Matrix VS0;
	VS0 =	V+S;	
	Matrix VS = chol2inv(VS0);
	Matrix CI(m,m);
	SymmetricMatrix Vsym(m);
	Vsym << VS;
	// -----------
	rwishart(nu+n, Vsym, &CI, IW); // IW <-> Sigma
	// -----------
	// B = Btilde + IR%*%matrix(rnorm(m*k),ncol=m)%*%t(rwout$CI)
	Real rn[m*k];
	for(int i=0 ; i < m*k; i++) {
		rn[i] = rnorm(0.,1.);
	}
	Matrix Brn(k,m);
	Brn << rn;
	*B = Btilde + (IR*(Brn*CI.t())); // B <-> Delta
}

//----------------------------------------------------------------------

