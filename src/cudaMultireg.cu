
// Interface with R

#define WANT_STREAM									// include.h will get stream fns
#define WANT_MATH										// include.h will get math fns
																		// newmatap.h will get include.h
#include "newmatap.h"								// need matrix applications

#include "newmatio.h"								// need matrix output routines

#ifdef use_namespace
using namespace NEWMAT;							// access NEWMAT namespace
#endif

#include "utilsFuncs.h" // function prototypes
// To be linked with utilsNewmat.cc utilsRmultireg.cc
// #include "utilsNewmat.cc"
// #include "utilsRmultireg.cc"

// includes, project
#include <cutil_inline.h>
#define BLOCK 64		// specify CUDA block size
#define XDIM 5			//  max number of reg. coeffs.
#define MDIM XDIM*XDIM

#define OBSDIM 128 	// max length of time series obs.; must be >= nobs
#define REGDIM 4096 // max. n.of regressions; must be >= nreg

// #define EMU 1
#define TIMER 1

// Preprocessed input option data
// allocate constant memory

static __device__ __constant__ float d_X[OBSDIM*XDIM];
static __device__ __constant__ float d_XpX[MDIM];
static __device__ __constant__ float d_Abeta[MDIM];
static __device__ __constant__ float d_ssq[REGDIM];

// includes, kernels
#include "cudaMultiregKnr.cu"

extern "C" {

void cudaMultireg(float* y, float* X, int* pnue, int* pnreg, int* pnobs, int* pnvar, int* pR, int* pkeep, float* pVbetadraw, float* pDeltadraw, float* pBetadraw, float* pTaudraw);

}

void cudaMultireg(float* y, float* X, int* pnue, int* pnreg, int* pnobs, int* pnvar, int* pR, int* pkeep, float* pVbetadraw, float* pDeltadraw, float* pBetadraw, float* pTaudraw)
{
	cudaSetDevice( cutGetMaxGflopsDeviceId() );
	//--------------------------------------------------------------------
	int nreg = (*pnreg);
	int nobs = (*pnobs);
	int nvar = (*pnvar);
	int nu = (*pnue) + nvar;
	int R = *pR;
	int keep = *pkeep; 
	int seed;  // seed passed to the kernel
	// int seed = 1234;

	if(nreg > REGDIM) { cout << "ERROR: REGDIM exceeded" ; return; } 
	if(nobs > OBSDIM) { cout << "ERROR: OBSDIM exceeded" ; return; } 
	if(nvar > XDIM) 	{ cout << "ERROR: XDIM exceeded" ; return; } 

// -------------------------------
// Initial MCMC Parameters 
	Matrix XM0(nvar, nobs); // !!! transposed dim - R column order
	XM0 << X; 
	Matrix XM = XM0.t();
	Matrix XpX = crossprod(XM, XM);
	Real* pXpX = XpX.data();
//
	DiagonalMatrix D(nvar); D =	1;
	SymmetricMatrix Vbeta(nvar);
	Vbeta = 0;
	Vbeta.inject(D);

	Matrix Betabar(nreg, nvar); Betabar = 0;
	Matrix Abeta(nvar,nvar); Abeta = 0;
	Matrix Z(nreg,1); Z = 1; // simplest case here 
	int nz = Z.ncols();
	Matrix Delta(nz,nvar); Delta = 0.; // to be updated 
	Matrix Deltabar(nz,nvar); Deltabar = 0.; // always zero (in all iterations)  in my  examples
	DiagonalMatrix Da(nz);
 	Da=0.01;
	SymmetricMatrix A(nz); A = 0;
	A.inject(Da);

	DiagonalMatrix Dv(nvar);
	Dv = nu;
	SymmetricMatrix V(nvar); V = 0;
	V.inject(Dv);
//	
	Real* betabar;
	Real* abeta;
//
	Matrix Y0( nreg,nobs); // !!!  transposed dim : R column order 
	Y0 << y;
	Matrix Y = Y0.t(); // use row order
//
  ColumnVector yi;
	Real ivar;
	Real tau[nreg];
	Real ssq[nreg];
	for(int i=0; i < nreg; i++) {
    yi = Y.column(i+1);
		varnr(yi.data(), nobs, &ivar); 
 		tau[i] = ivar;
 		ssq[i] = ivar;
	}
	//--------------------
	// MCMC simulation
	int mkeep, px;
	int dimVbetadraw = nvar*nvar;
	Matrix Vaux(nvar,nvar); 
	int dimDeltadraw = nz*nvar;
	int dimtaudraw = nreg;
	int dimBetadraw = nreg * nvar;
 	// used to export matricial MCMC results in R-column order 
	Matrix TBetabar(nreg,nvar);
	Matrix TDelta(nz,nvar);
	//--------------------------------------------------------------------
	// setup execution parameters
	int nthreads, nblocks;
	div_t d = div((*pnreg), BLOCK);
	if(*pnreg <= BLOCK) {
		nblocks = 1;
		nthreads = (*pnreg);
	} else {
 		// not necessarily a multiple of block size
		nblocks = int(ceil(float(*pnreg)/BLOCK));
		nthreads = BLOCK;
	}
	dim3 dGrid = nblocks; 
	dim3 dBlock = nthreads;
	//--------------------------------------------------------------------
#ifdef TIMER
	printf("n. of required threads = %d\n",(*pnreg)); 
	printf("dGrid = %d \t dBlock = %d \n", nblocks, nthreads);
	// create and start timer

	unsigned int timer = 0;
	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));
#endif 
	//--------------------------------------------------------------------
	// memcopy to constant memory
	unsigned int mem_size_xregdim = sizeof(float) * nobs * nvar;
	unsigned int mem_size_mdim = sizeof(float) * nvar * nvar;
	unsigned int mem_size_regdim = sizeof(float) * nreg;
  cutilSafeCall( cudaMemcpyToSymbol( d_X, X, mem_size_xregdim) );
  cutilSafeCall( cudaMemcpyToSymbol( d_XpX, pXpX, mem_size_mdim) );
  cutilSafeCall( cudaMemcpyToSymbol( d_ssq, ssq, mem_size_regdim) );
	//--------------------------------------------------------------------
	// betabar/betadraw : allocate device memory
	unsigned int size_BETABAR = XDIM * REGDIM;
	unsigned int mem_size_BETADIM = sizeof(float) * size_BETABAR;
	float* d_betabar;
	cutilSafeCall(cudaMalloc((void**) &d_betabar, mem_size_BETADIM));
	unsigned int mem_size_betadim = sizeof(float) * nvar * nreg;

	// allocate device memory
	float* d_tau;
	unsigned int mem_size_TAUDIM = sizeof(float) * REGDIM;
	cutilSafeCall(cudaMalloc((void**) &d_tau, mem_size_TAUDIM));

	// copy host memory to device
	unsigned int mem_size_taudim = sizeof(float) * nreg;

	// allocate device memory
	unsigned int mem_size_YDIM = sizeof(float) * OBSDIM * REGDIM;
	float* d_y;
	cutilSafeCall(cudaMalloc((void**) &d_y, mem_size_YDIM));

	// copy host memory to device
	unsigned int mem_size_ydim = sizeof(float) * nobs * nreg;
	cutilSafeCall(cudaMemcpy(d_y, y, mem_size_ydim, cudaMemcpyHostToDevice) );

	// -------------------------------
	// MCMC simulation
	//
	cout << "Processing " << R << " iterations:\t '.' = 100 iterations" << endl; 
	
	for(int rep=1; rep <= R; rep++) {
	
		Abeta = chol2inv(Vbeta);
		abeta = Abeta.data();
		Betabar = Z * Delta;
		betabar = Betabar.data();
	
		// -------------------------------
		// copy host memory to device at each iteration
	
		cutilSafeCall(cudaMemcpy(d_betabar, betabar, mem_size_betadim, cudaMemcpyHostToDevice) );
		cutilSafeCall(cudaMemcpy(d_tau, tau, mem_size_taudim, cudaMemcpyHostToDevice) );
	  cutilSafeCall( cudaMemcpyToSymbol( d_Abeta, abeta, mem_size_mdim) );
	
		// -------------------------------
		// Run regression kernel
	
		seed = rand();
		// printf("kernel seed = %d\n", seed);
	
		cudaruniregNRK<<< dGrid, dBlock >>>(d_betabar, d_tau, d_y, nu, nreg, nobs, nvar, seed);
	
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");

		// -------------------------------
		// Update values : betadraw , tau
		// copy betadraw from device to host
	
		cutilSafeCall(cudaMemcpy(betabar, d_betabar, mem_size_betadim, cudaMemcpyDeviceToHost) );
		// copy tau/sigmasqdraw from device to host
		cutilSafeCall(cudaMemcpy(tau, d_tau, mem_size_taudim, cudaMemcpyDeviceToHost) );
	
		Betabar << betabar; 
	
		// -------------------------------
		// Apply rmultireg
		// Returned Values : Delta , Vbeta
	
	 	rmultireg(Betabar,Z,Deltabar,A,nu,V, &Delta, &Vbeta); // Delta <-> B ; Vbeta <-> Sigma
	
		// -------------------------------
		// keep MCMC draws
	
	  if(rep%keep == 0) { 
			// cout << "\n*\n";
	  	mkeep=rep/keep;
			// Symmetric values = do not transpose;  dimVbetadraw = nvar*nvar;
			Vaux.inject(Vbeta); //!!! required due to SymmetricMatrix storage scheme.
			px = (mkeep-1) * dimVbetadraw;
			newmat_block_copy(dimVbetadraw, Vaux.data(), pVbetadraw+px);	
			// dimDeltadraw = nz*nvar;
			if(nz == 1) { // vector do not transpose
				px = (mkeep-1) * dimDeltadraw;
				newmat_block_copy(dimDeltadraw, Delta.data(), pDeltadraw+px);	
			}
			else { // use transpose
				px = (mkeep-1) * dimDeltadraw;
				TDelta = Delta.t();
				newmat_block_copy(dimDeltadraw, TDelta.data(), pDeltadraw+px);	
			}
			// vector: dimtaudraw = nreg;
			px = (mkeep-1) * dimtaudraw;
			newmat_block_copy(dimtaudraw, tau, pTaudraw+px);	
			// matrix as simulated array
			px = (mkeep-1) * dimBetadraw;
			TBetabar = Betabar.t(); // use R column order
			newmat_block_copy(dimBetadraw, TBetabar.data(), pBetadraw+px);	
		}
	
		if(rep%100 == 0)
			cout.flush() << ".";
	
	}
	
	cout << endl;

	// -------------------------------

#ifdef TIMER
	// stop and destroy timer
	cutilCheckError(cutStopTimer(timer));
	printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
	cutilCheckError(cutDeleteTimer(timer));
#endif

	// clean up memory

	cutilSafeCall(cudaFree(d_y));
	cutilSafeCall(cudaFree(d_tau));
	cutilSafeCall(cudaFree(d_betabar));

	cudaThreadExit();

}


