//$Id: d_rngMT.cu,v 1.5 2010/05/15 14:39:39 afs Exp $

/* 
   A C-program for MT19937-64 (2004/9/29 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.
   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          
*/

// Modified for CUDA by A. Ferreira da Silva (C) May 2010


namespace MT312_rng {

// typedef unsigned long long ULLINT; 

class __align__(16) Rng {

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */

	/* The array for the state vector */
	unsigned long long mt[NN]; 
	/* mti==NN+1 means mt[NN] is not initialized */
	int mti;
	ulint seed;
public:
	__device__ Rng() : seed(123456789LL) {}
	__device__ Rng(unsigned long long iseed) : seed(iseed) {
		mti=NN+1; 
		init_genrand64(seed);
	} 
	/* initializes mt[NN] with a seed */
	__device__ void init_genrand64(unsigned long long seed) {
    mt[0] = seed;
    for (mti=1; mti<NN; mti++) 
        mt[mti] =  (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
	}
	/* generates a random number on [0, 2^64-1]-interval */
	__device__ unsigned long long genrand64_int64(void) {
    int i;
    unsigned long long x;
    unsigned long long mag01[2]={0ULL, MATRIX_A};
    if (mti >= NN) { /* generate NN words at one time */
        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (mti == NN+1) 
            init_genrand64(5489ULL); 
        for (i=0;i<NN-MM;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        for (;i<NN-1;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        x = (mt[NN-1]&UM)|(mt[0]&LM);
        mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        mti = 0;
    }
    x = mt[mti++];
    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);
    return x;
	}
	/* generates a random number on [0, 2^63-1]-interval */
	__device__ long long genrand64_int63(void) {
	    return (long long)(genrand64_int64() >> 1);
	}
	/* generates a random number on [0,1)-real-interval */
	__device__ real dev(void) {
	    return (genrand64_int64() >> 11) * (1.0/9007199254740992.0);
	}
#undef NN 
#undef MM 
#undef MATRIX_A 
#undef UM 
#undef LM 
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
