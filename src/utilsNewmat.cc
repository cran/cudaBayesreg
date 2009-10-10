
#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
                                     // newmatap.h will get include.h
#include "newmatap.h"                // need matrix applications

#include "newmatio.h"                // need matrix output routines


#ifdef use_namespace
using namespace NEWMAT;              // access NEWMAT namespace
#endif

#include "utilsFuncs.h"    

//===================================================================
// Newmat utils

ReturnMatrix upperTri(const int m, Real* vals)
{
	// required dim of val vector:	m*(m+1)/2-m
	Matrix UT(m,m);
	for(int i=1, k=0; i<=m; i++) for(int j=1; j<=m; j++) 
			if(j <= i) UT(i,j) = 0.; else UT(i,j) = vals[k++];
	UT.release(); return UT;
}

ReturnMatrix lowerTri(const int m, Real* vals)
{
	// required dim of val vector:	m*(m+1)/2-m
	Matrix LT(m,m);
	for(int j=1, k=0; j<=m; j++) for(int i=1; i<=m; i++) 
			if(j >= i) LT(i,j) = 0.; else LT(i,j) = vals[k++];
	LT.release(); return LT;
}


void diagSub(Matrix* T, Real* val)
{
	// required dim of val vector:	m
	int m = T->nrows();
	Real* a = T->data();
	for(int i=0, k=0; i < m; i++) 
		*(a+i*m+i) = val[k++];
}

// U must be UpperTriangular
ReturnMatrix backsolveU(const Matrix &U)
{
	int n = U.ncols();
	DiagonalMatrix D(n); D = 1;     
	ColumnVector ci; 
	Matrix* IR = new Matrix(n,n);
	for(int i=0 ; i < n; i++) {
		ci = U.i() * D.column(i+1);
		IR->column(i+1) =ci;
	}
	IR->release_and_delete(); return *IR;
}


ReturnMatrix backsolve(const Matrix &W)
{
	int n = W.ncols();
	SymmetricMatrix CW;
 	CW << W.t() * W;
	DiagonalMatrix D(n); D = 1;     
  UpperTriangularMatrix U = Cholesky(CW).t();
	ColumnVector ci; 
	Matrix* IR = new Matrix(n,n);
	for(int i=0 ; i < n; i++) {
		ci = U.i() * D.column(i+1);
		IR->column(i+1) =ci;
	}
	IR->release_and_delete(); return *IR;
}

ReturnMatrix crossprod(const Matrix &A, const Matrix &B)
{
	// int n = A.ncols();
	Matrix C= A.t() * B;
	C.release(); return C;
}

ReturnMatrix tcrossprod(const Matrix &A, const Matrix &B)
{
	// int n = A.ncols();
	Matrix C= A * B.t();
	C.release(); return C;
}

ReturnMatrix tcrossprod2(const Matrix &A, const Matrix &B)
{
	int n = A.ncols();
	Matrix* C = new Matrix(n,n);
	*C = A * B.t();
	C->release_and_delete(); return *C;
}

// ReturnMatrix chol2inv(SymmetricMatrix &A)
ReturnMatrix chol2inv(const Matrix &A)
{
	Matrix C = A.i();
	C.release(); return C;
}

