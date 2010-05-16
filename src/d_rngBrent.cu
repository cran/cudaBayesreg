// $Id: d_rngBrent.cu,v 1.4 2010/05/15 14:39:39 afs Exp $
//
// Copyright (C) 2004, 2006, 2008 R. P. Brent.    
// xorgens.c version 3.05, R. P. Brent, 20080920.
//
// Original RNG code by R. P. Brent
// R. P. Brent, Some uniform and normal random number generators:
//   xorgens version 3.04, 28 June 2006.
// http://wwwmaths.anu.edu.au/~brent/random.html.
//
// Modified for CUDA by A. Ferreira da Silva (C) May 2010
//

namespace Brent_rng {

class __align__(16) Rng {

#define UINT64 (sizeof(ulint)>>3)
#define UINT32 (1 - UINT64) 
#define REAL64 (sizeof(real)>>3)
#define REAL32 (1 - REAL64)
#define wlen (64*UINT64 +  32*UINT32)
#define r    (64*UINT64 + 128*UINT32)
#define s    (53*UINT64 +  95*UINT32)
#define a    (33*UINT64 +  17*UINT32)
#define b    (26*UINT64 +  12*UINT32)
#define c    (27*UINT64 +  13*UINT32)
#define d    (29*UINT64 +  15*UINT32)
#define ws   (27*UINT64 +  16*UINT32) 
#define sr (11*REAL64 +(40*UINT64 + 8*UINT32)*REAL32)
#define ss ((53*UINT64 + 21*UINT32)*REAL64 + 24*REAL32)
#define SCALE ((real)1/(real)((ulint)1<<ss)) 
#define SC32  ((real)1/((real)65536*(real)65536)) 
  ulint w, weyl, zero , x[r];
	ulint seed;
	real res;
  int i;
public:
	__device__ Rng() : zero(0), seed(1), i(-1) {}
	__device__ Rng(ulint iseed) : zero(0), seed(iseed), i(-1) { 
		xor4096r(seed);
	}
	__device__ ulint xor4096i(ulint seed) {
	  ulint t, v;
	  ulint k;
	  if ((i < 0) || (seed != zero)) {
	    if (UINT32) 
	      weyl = 0x61c88647;
	    else 
	      weyl = ((((ulint)0x61c88646)<<16)<<16) + (ulint)0x80b583eb;
	    v = (seed!=zero)? seed:~seed; 
	    for (k = wlen; k > 0; k--) {  
	      v ^= v<<10; v ^= v>>15;    
	      v ^= v<<4;  v ^= v>>13;   
	      }
	    for (w = v, k = 0; k < r; k++) {
	      v ^= v<<10; v ^= v>>15; 
	      v ^= v<<4;  v ^= v>>13;
	      x[k] = v + (w+=weyl);                
	      }
	    for (i = r-1, k = 4*r; k > 0; k--) { 
	      t = x[i = (i+1)&(r-1)];   t ^= t<<a;  t ^= t>>b; 
	      v = x[(i+(r-s))&(r-1)];   v ^= v<<c;  v ^= v>>d;          
	      x[i] = t^v;       
	      }
	    }
	  t = x[i = (i+1)&(r-1)];          
	  v = x[(i+(r-s))&(r-1)];         
	  t ^= t<<a;  t ^= t>>b;         
	  v ^= v<<c;  v ^= v>>d;        
	  x[i] = (v ^= t);             
	  w += weyl;                  
	  return (v + (w^(w>>ws)));  
	
	}
	__device__ void xor4096r(ulint seed) {
	  res = (real)0; 
	  while (res == (real)0)            
	  {                                 
	    res = (real)(xor4096i(seed)>>sr);
	    seed = (ulint)0;                  
	    if (UINT32 && REAL64)          
	      res += SC32*(real)xor4096i(seed);
	  }
	}
	__device__ real dev() {
	  res = (real)0; 
	  while (res == (real)0)              
	  {                                   
	    res = (real)(xor4096i(0)>>sr);  
	    if (UINT32 && REAL64)             
	      res += SC32*(real)xor4096i(0);
	  }
	  return (SCALE*res);               
	}

#undef wlen
#undef r
#undef s
#undef a
#undef b
#undef c
#undef d
#undef ws 
#undef SC32
#undef SCALE
#undef sr
#undef ss
#undef UINT64
#undef UINT32
#undef REAL64
#undef REAL32
};

class __align__(16) rngNormal : Rng {
	real BM_norm_keep;
public:
	__device__ rngNormal() : Rng(), BM_norm_keep(0.) {}
	__device__ rngNormal(ulint iseed) : Rng(iseed), BM_norm_keep(0.) {}
	__device__ real dev() {
		if(BM_norm_keep != 0.0) { 
	    real sbm = BM_norm_keep;
	    BM_norm_keep = 0.0;
	    return sbm;
		} else {
	    real theta = 2 * M_PI * Rng::dev();
	    real R = sqrt(-2 * log(Rng::dev())) + 10*DBL_MIN;
	    BM_norm_keep = R * sin(theta);
	    return R * cos(theta);
		}
	}
};

class __align__(16) rngGamma : rngNormal {
	real alpha, alphain, beta;
	real dg,cg;
public:
	__device__ rngGamma() {}
	__device__ rngGamma(real alpha, real beta, ulint iseed)
		: rngNormal(iseed), alpha(alpha), alphain(alpha), beta(beta) {
		if (alpha <= 0.) return;
		if (alpha < 1.) alpha += 1.;
		dg = alpha-1./3.;
		cg = 1./sqrt(9.*dg);
	}
	__device__ real square(real x) { return (x == 0.0) ? 0.0 : x*x; }
	__device__ real dev() {
		real u,v,x;
		do {
			do {
				x = rngNormal::dev();
				v = 1. + cg*x;
			} while (v <= 0.);
			v = v*v*v;
			u = rngNormal::dev();
		} while (u > 1. - 0.331*square(square(x)) &&
			log(u) > 0.5*square(x) + dg*(1.-v+log(v))); 
		if (alpha == alphain) return dg*v/beta;
		else { 
			do u=rngNormal::dev(); while (u == 0.);
			return powf(u,1./alphain)*dg*v/beta;
		}
	}
	__device__ real d_rnorm() { return rngNormal::dev(); }
	__device__ real d_rchisq() { return rngGamma::dev(); }
};

}
