//$Id: cudaMultiregKnr1.cu,v 1.4 2010/05/15 14:39:39 afs Exp $
//
// !!! Processing in mycudamath functions assumes column order for compatibility with R
//

#include "d_rngBrent.cu"

namespace Kbrent {

using namespace Brent_rng;

__global__ void
cudaruniregNRK(float* d_betabar, float* tau, float* y, int nu, int nreg, int nobs, int m, int seed)
{
	const int ti = blockIdx.x * blockDim.x + threadIdx.x;
	if(ti >= nreg) return;
	const float df = nu+nobs; 
	const ulint seedti = (ulint)((seed >> 1)+ti);
	rngGamma drng(df / 2.0f, 0.5f, seedti);
//
	float* X = d_X;
	float* XpX = d_XpX;
	float* A = d_Abeta;
	float* ssq = d_ssq;
	const int mxm = m*m;
//
 	const float	sigmasq = tau[ti];
	//----------------------------
  // IR=backsolve(chol(XpX/sigmasq+A),diag(k))
	float IR[MDIM];
	{
		float tmp0[MDIM];
		for(int i=0; i < mxm; i++) 
			tmp0[i] = XpX[i] / sigmasq + A[i];
		mdgbacksolve(tmp0, &m,  IR);
	}
	//----------------------------
	// Xpy
	float Xpy[XDIM];
	float* yblock = &y[ti*nobs];

	mvtcrossp(X, yblock, Xpy, &nobs, &m);
	//----------------------------
	// btilde=crossprod(t(IR))%*%(Xpy/sigmasq+A%*%betabar)
	float* betabar = &d_betabar[ti*m];
	float btilde[XDIM];
	{
		float tmp1[XDIM];
		mvprod(A,betabar, tmp1, &m, &m);
		// (Xpy/sigmasq+A%*%betabar)
		for (int i=0; i<m; i++)
			tmp1[i] = Xpy[i] / sigmasq + tmp1[i]; 
		// crossprod(t(IR))
		float cIR[MDIM];
		mtcrossp(IR, IR, cIR, &m);
		mvprod(cIR, tmp1, btilde, &m, &m);
	}
	//----------------------------
  // beta = btilde + IR%*%rnorm(k)
	// Update betabar
	float beta[XDIM];
	{
		float tmp1[XDIM];
		for(int i=0; i < m; i++) tmp1[i]=drng.d_rnorm();
	 	// d_rnorm(&drng, m, tmp1);
		mvprod(IR, tmp1, beta, &m, &m);
	  for (int i=0; i < m; i++) 
			beta[i] = beta[i] + btilde[i]; 
	}
	//----------------------------
  // res=y-X%*%beta
  // s=t(res)%*%res
  // sigmasq=(nu*ssq + s)/rchisq(1,nu+n)
	float s;
	float resid[OBSDIM];
	mvprod(X, beta, resid, &nobs, &m);
	for(int i=0; i < nobs; i++) 
		resid[i] = yblock[i] - resid[i];
	vprod(resid, resid, &s, &nobs);
	float rchi;
	rchi = drng.d_rchisq();
	// d_rchisq(&drng, 1, &rchi);
	// d_rchisq(1, nu+nobs, 1234, &rchi);
	//----------------------------
	// Results
	tau[ti] = (nu*ssq[ti] + s)/rchi;
	//	__syncthreads();
	int ix=ti*m;
	for(int i=0; i < m; i++) {
		d_betabar[ix+i] = beta[i];
	}
}

}
