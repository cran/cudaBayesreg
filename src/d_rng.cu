// Copyright (C) 2004: R B Davies

#include <climits>
#include <cstdio>
#include <ctime>
#include <math.h>

#define Ullint unsigned long long int
#define Ulint unsigned long int

// #define REAL double
#define REAL float

#define Uint unsigned int
#define Int int

__device__ static float sqrarg;
#define SQUARE(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

#define UINT32_MAX             (4294967295U)
#define UINT64_MAX             (__UINT64_C(18446744073709551615))

Ulint hashseed( time_t t, clock_t c )
{
	// Get a Ulint from t and c
	// Better than Ulint(x) in case x is floating point in [0,1]
	// Based on code by Lawrence Kirby (fred@genesis.demon.co.uk)
	static Ulint differ = 0;  // guarantee time-based seeds will change
	Ulint h1 = 0;
	unsigned char *p = (unsigned char *) &t;
	for( size_t i = 0; i < sizeof(t); ++i )
	{
		h1 *= UCHAR_MAX + 2U;
		h1 += p[i];
	}
	Ulint h2 = 0;
	p = (unsigned char *) &c;
	for( size_t j = 0; j < sizeof(c); ++j )
	{
		h2 *= UCHAR_MAX + 2U;
		h2 += p[j];
	}
	return ( h1 + differ++ ) ^ h2;
}

Ulint rseed()
{
	// Seed the generator with an array from /dev/urandom if available
	// Otherwise use a hash of time() and clock() values
	// First try getting an array from /dev/urandom
	FILE* urandom = fopen( "/dev/urandom", "rb" );
	if( urandom )
	{
		int N = 1;
		Ulint bigSeed;
		register Ulint *s = &bigSeed;
		register int i = N;
		register bool success = true;
		while( success && i-- )
			success = fread( s++, sizeof(Ulint), 1, urandom );
		fclose(urandom);
		if( success ) { // seed( bigSeed, N );
			return bigSeed; }
	}
	// Was not successful, so use time() and clock() instead
	// seed( hash( time(NULL), clock() ) );
	return hashseed( time(NULL), clock() );
}

__device__ struct __align__(16) RNG {
	Ullint u,v,w;
	__device__ RNG() : v(4101842887655102017LL), w(1) {}
	__device__ RNG(Ullint j) : v(4101842887655102017LL), w(1) {
		u = j ^ v; getint64();
		v = u; getint64();
		w = v; getint64();
	}
	__device__ inline Ullint getint64() {
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
		w = 4294957665U*(w & 0xffffffff) + (w >> 32);
		Ullint x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
		return (x + v) ^ w;
	}
	__device__ inline REAL getdoub() { return 5.42101086242752217E-20 * getint64(); }
	__device__ inline Uint getint32() { return (Uint)getint64(); }
	__device__ inline Int  getint1n(Uint n) { return(1 + getint64() % (n-1)); }
};
	
struct __align__(16) rngNormal : RNG {
	REAL mu,sig;
	REAL storedval;
	__device__ rngNormal() : RNG(), mu(0), sig(1), storedval(0.) {}
	__device__ rngNormal(REAL mmu, REAL ssig, Ullint i)
	: RNG(i), mu(mmu), sig(ssig), storedval(0.) {}
	 __device__ inline REAL dev() {
		REAL v1,v2,rsq,fac;
		if (storedval == 0.) {
			do {
				v1=2.0*getdoub()-1.0;
				v2=2.0*getdoub()-1.0;
				rsq=v1*v1+v2*v2;
			} while (rsq >= 1.0 || rsq == 0.0);
			fac=sqrt(-2.0*log(rsq)/rsq);
		 	storedval = v1*fac;
			return mu + sig*v2*fac;
		} else {
				fac = storedval;
				storedval = 0.;
				return mu + sig*fac;
		}
	}
};


struct __align__(16) rngGamma : rngNormal {
	REAL alph, oalph, bet;
	REAL a1,a2;
	 __device__ rngGamma(REAL aalph, REAL bbet, Ullint i)
	: rngNormal(0.,1.,i), alph(aalph), oalph(aalph), bet(bbet) {
		if (alph <= 0.) return;
		if (alph < 1.) alph += 1.;
		a1 = alph-1./3.;
		a2 = 1./sqrt(9.*a1);
	}
	 __device__ inline REAL dev() {
		REAL u,v,x;
		do {
			do {
				x = rngNormal::dev();
				v = 1. + a2*x;
			} while (v <= 0.);
			v = v*v*v;
			u = getdoub();
		} while (u > 1. - 0.331*SQUARE(SQUARE(x)) &&
			log(u) > 0.5*SQUARE(x) + a1*(1.-v+log(v))); 
		if (alph == oalph) return a1*v/bet;
		else { 
			do u=getdoub(); while (u == 0.);
			return powf(u,1./oalph)*a1*v/bet;
		}
	}
};

__device__ inline void d_rnorm(rngNormal* nd,  int n, float mu, float sig, REAL* res)
{
	for(int i=0; i<n; i++) 
		res[i] = nd->dev();
}
	
__device__ inline void d_rchisq(rngGamma* chi, int n, REAL* res)
{
	for(int i=0; i<n; i++) 
		res[i] = chi->dev();
}

