//$Id: cudaMultireg.cu,v 1.10 2010/05/16 10:29:43 afs Exp afs $

#include <float.h>

#define WANT_STREAM    
#define WANT_MATH    
#include "newmatap.h"  
#include "newmatio.h"

#ifdef use_namespace
using namespace NEWMAT;
#endif

#include "utilsFuncs.h"
// #include <cutil_inline.h>
#define TIMER 1

typedef float real;
typedef unsigned long ulint; 

// #define BLOCK 64    // specify CUDA block size
#define BLOCK 128    // specify CUDA block size
#define XDIM 5      //  max number of reg. coeffs.
#define MDIM XDIM*XDIM
// #define OBSDIM 96   // max length of time series obs.; must be >= nobs
#define OBSDIM 128   // max length of time series obs.; must be >= nobs
#define REGDIM 4096 // max. n.of regressions; must be >= nreg


// Preprocessed input option data
// allocate constant memory

static __device__ __constant__ float d_X[OBSDIM*XDIM];
static __device__ __constant__ float d_XpX[MDIM];
static __device__ __constant__ float d_Abeta[MDIM];
static __device__ __constant__ float d_ssq[REGDIM];

#include "utilsSeeds.cc"
#include "mycudamath.cu"
#include "cudaMultiregKnr0.cu"
#include "cudaMultiregKnr1.cu"
#include "cudaMultiregKnr2.cu"

extern "C" {

void cudaMultireg(float* y, float* X, float* pZ, float* pDeltabar, int* pnz, int* pnue, int* pnreg, int* pnobs, int* pnvar, int* pR, int* pkeep, int* prng, float* pVbetadraw, float* pDeltadraw, float* pBetadraw, float* pTaudraw);

}

void cudaMultireg(float* y, float* X, float* pZ, float* pDeltabar, int* pnz, int* pnue, int* pnreg, int* pnobs, int* pnvar, int* pR, int* pkeep, int* prng, float* pVbetadraw, float* pDeltadraw, float* pBetadraw, float* pTaudraw)
{
  // cudaSetDevice( cutGetMaxGflopsDeviceId() );
  //--------------------------------------------------------------------
  int nreg = (*pnreg);
  int nobs = (*pnobs);
  int nvar = (*pnvar);
  int nue = (*pnue);
  int nu = (*pnue) + nvar;
  int nz = (*pnz);
  int R = *pR;
  int keep = *pkeep; 
  int seed1, seed2;  
	srand(rseed());

  if(nreg > REGDIM) { cout << "ERROR: REGDIM exceeded" ; return; } 
  if(nobs > OBSDIM) { cout << "ERROR: OBSDIM exceeded" ; return; } 
  if(nvar > XDIM)   { cout << "ERROR: XDIM exceeded" ; return; } 

// -------------------------------
// Initial MCMC Parameters 
  Matrix XM0(nvar, nobs); // !!! transposed dim - R column order
  XM0 << X; 
  Matrix XM = XM0.t();
  Matrix XpX = crossprod(XM, XM);
  float* pXpX = XpX.data();
//
  DiagonalMatrix D(nvar); D =  1.0f;
  SymmetricMatrix Vbeta(nvar);
  Vbeta = 0.f;
  Vbeta.inject(D);
//
  Matrix Betabar(nreg, nvar); Betabar = 0.f;
  Matrix Abeta(nvar,nvar); Abeta = 0.f;
  Matrix Z0(nz, nreg);
  Z0 << pZ;
  Matrix Z = Z0.t();
   // Deltabar(nz,nvar); always 0.f (in all iterations) if not explicitly specified as prior
  Matrix Deltabar0(nvar,nz);
  Deltabar0 << pDeltabar;
  Matrix Deltabar = Deltabar0.t();
  Matrix Delta(nz,nvar); Delta = 0.f; // to be updated 
//
  DiagonalMatrix Da(nz);
  Da=0.01f;
  SymmetricMatrix A(nz); A = 0.f;
  A.inject(Da);
//
  DiagonalMatrix Dv(nvar);
  Dv = nu;
  SymmetricMatrix V(nvar); V = 0.f;
  V.inject(Dv);
//  
  float* betabar;
  float* abeta;
//
  Matrix Y0( nreg,nobs); // !!!  transposed dim : R column order 
  Y0 << y;
  Matrix Y = Y0.t(); // use row order
//
  ColumnVector yi;
  float ivar;
  float tau[nreg];
  float ssq[nreg];
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
  printf("n. of required threads = %d\n",(*pnreg)); 
  printf("dGrid = %d \t dBlock = %d \n", nblocks, nthreads);
  //--------------------------------------------------------------------
  // memcopy to constant memory
  unsigned int mem_size_xregdim = sizeof(float) * nobs * nvar;
  unsigned int mem_size_mdim = sizeof(float) * nvar * nvar;
  unsigned int mem_size_regdim = sizeof(float) * nreg;

  cudaMemcpyToSymbol( d_X, X, mem_size_xregdim) ;
  cudaMemcpyToSymbol( d_XpX, pXpX, mem_size_mdim) ;
  cudaMemcpyToSymbol( d_ssq, ssq, mem_size_regdim) ;
  //--------------------------------------------------------------------
  // betabar/betadraw : allocate device memory
  unsigned int size_BETABAR = XDIM * REGDIM;
  unsigned int mem_size_BETADIM = sizeof(float) * size_BETABAR;
  float* d_betabar;
  cudaMalloc((void**) &d_betabar, mem_size_BETADIM);
  unsigned int mem_size_betadim = sizeof(float) * nvar * nreg;

  // allocate device memory
  float* d_tau;
  unsigned int mem_size_TAUDIM = sizeof(float) * REGDIM;
  cudaMalloc((void**) &d_tau, mem_size_TAUDIM);

  // copy host memory to device
  unsigned int mem_size_taudim = sizeof(float) * nreg;

  // allocate device memory
  unsigned int mem_size_YDIM = sizeof(float) * OBSDIM * REGDIM;
  float* d_y;
  cudaMalloc((void**) &d_y, mem_size_YDIM);

  // copy host memory to device
  unsigned int mem_size_ydim = sizeof(float) * nobs * nreg;
  cudaMemcpy(d_y, y, mem_size_ydim, cudaMemcpyHostToDevice) ;

  // MODX1: copy once
  cudaMemcpy(d_tau, tau, mem_size_taudim, cudaMemcpyHostToDevice);

  // -------------------------------
  // MCMC simulation
  //
  cout << "Processing " << R << " iterations:\t '.' = 100 iterations" << endl; 

  // -------------------------------
#ifdef TIMER
  cudaEvent_t start;
  cudaEvent_t end;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start,0);


/*
  unsigned int timer = 0;
  cutCreateTimer(&timer);
  cutStartTimer(timer);
*/
#endif
  // -------------------------------

  for(int rep=1; rep <= R; rep++) {

    Abeta = chol2inv(Vbeta); abeta = Abeta.data();
    Betabar = Z * Delta; betabar = Betabar.data();

    // -------------------------------
    // copy host memory to device at each iteration
  
    cudaMemcpy(d_betabar, betabar, mem_size_betadim, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_Abeta, abeta, mem_size_mdim);
  
    // -------------------------------
    // Run regression kernel
  
//   cudaThreadSynchronize() ;


  	// seed1 = rseed();
  	seed1 = rand();

    switch(*prng) {
    case 0:
    default: {
      // seed2 = rseed();
      seed2 = rand();
      Kmars::cudaruniregNRK<<< dGrid, dBlock >>>(d_betabar, d_tau, d_y, nue,
         nreg, nobs, nvar, seed1, seed2); }
      break;
    case 1:
      Kbrent::cudaruniregNRK<<< dGrid, dBlock >>>(d_betabar, d_tau, d_y, nue,
         nreg, nobs, nvar, seed1);
      break;
    case 2:
      Kmt312::cudaruniregNRK<<< dGrid, dBlock >>>(d_betabar, d_tau, d_y, nue,
        nreg, nobs, nvar, seed1);
      break;
    }

    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl; 
        exit(-1);
    }

    // -------------------------------
    // Update values : betadraw , tau
    // copy betadraw from device to host

    cudaMemcpy(betabar, d_betabar, mem_size_betadim, cudaMemcpyDeviceToHost);

    // -------------------------------
    // Apply rmultireg
    // Returned Values : Delta , Vbeta
    // Using row-order

     rmultireg(Betabar,Z,Deltabar,A,nu,V, &Delta, &Vbeta); // Delta <-> B ; Vbeta <-> Sigma
  
    // -------------------------------
    // keep MCMC draws

    if(rep%keep == 0) { 
      // copy tau/sigmasqdraw from device to host
      cudaMemcpy(tau, d_tau, mem_size_taudim, cudaMemcpyDeviceToHost);

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

  cudaEventSynchronize(end);
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cout << "Processing time: " << elapsed_time << " milliseconds" << std::endl;

/*
  // stop and destroy timer
  cudaThreadSynchronize();
  cutStopTimer(timer);
  printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
  cutDeleteTimer(timer);
*/
#endif
  // -------------------------------

  // clean up memory
  cudaFree(d_y);
  cudaFree(d_tau);
  cudaFree(d_betabar);

  cudaThreadExit();

}


