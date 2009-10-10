
#ifndef _utilsFuncs_H
#define _utilsFuncs_H 1

extern "C" {

ReturnMatrix upperTri(const int m, Real* vals);
ReturnMatrix lowerTri(const int m, Real* vals);
void diagSub(Matrix* T, Real* val);
ReturnMatrix backsolve(const Matrix &W);
ReturnMatrix backsolveU(const Matrix &U);
ReturnMatrix crossprod(const Matrix &A, const Matrix &B);
ReturnMatrix tcrossprod(const Matrix &A, const Matrix &B);
ReturnMatrix tcrossprod2(const Matrix &A, const Matrix &B);
ReturnMatrix chol2inv(const Matrix &A);

void varnr(const Real data[], const int n, Real *var);
void rwishart(const int nu, const SymmetricMatrix V, Matrix* CI, SymmetricMatrix* IW);
void rmultireg(const Matrix& Y, const Matrix& X, const Matrix& Bbar, const SymmetricMatrix& A, const int nu, const Matrix& V, Matrix* B, SymmetricMatrix* IW);

}

#endif
