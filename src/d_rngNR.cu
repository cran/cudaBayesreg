
#include <math.h>
#include "d_nr3.h"


__device__ struct __align__(16) Ran {
	Ullong u,v,w;
	 __device__ Ran(Ullong j) : v(4101842887655102017LL), w(1) {
	// Constructor. Call with any integer seed (except value of v above).
		u = j ^ v; int64();
		v = u; int64();
		w = v; int64();
	}
	
	 __device__ inline Ullong int64() {
		// Return 64-bit random integer. See text for explanation of method.
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
		w = 4294957665U*(w & 0xffffffff) + (w >> 32);
		Ullong x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
		return (x + v) ^ w;
	}
	
	 __device__ inline Doub doub() { return 5.42101086242752217E-20 * int64(); }
	// Return random double-precision ﬂoating value in the range 0. to 1.

	 __device__ inline Uint int32() { return (Uint)int64(); }
	// Return 32-bit random integer.

	// a random integer between 1 and n (inclusive)
	__device__ inline Int int1n(Uint n) { return(1 + int64() % (n-1)); }
};
	


struct __align__(16) Normaldev_BM : Ran {
	// Structure for normal deviates.
	Doub mu,sig;
	Doub storedval;
	 __device__ Normaldev_BM(Doub mmu, Doub ssig, Ullong i)
	: Ran(i), mu(mmu), sig(ssig), storedval(0.) {}
	// Constructor arguments are mu, sigma, and a random sequence seed.
	 __device__ Doub dev() {
		// Return a normal deviate.
		Doub v1,v2,rsq,fac;
		if (storedval == 0.) {
			do {
				v1=2.0*doub()-1.0;
				v2=2.0*doub()-1.0;
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


struct __align__(16) Gammadev : Normaldev_BM {
// Structure for gamma deviates.
	Doub alph, oalph, bet;
	Doub a1,a2;
	 __device__ Gammadev(Doub aalph, Doub bbet, Ullong i)
	: Normaldev_BM(0.,1.,i), alph(aalph), oalph(aalph), bet(bbet) {
	// Constructor arguments are ˛, ˇ , and a random sequence seed.
//		if (alph <= 0.) throw("bad alph in Gammadev");
		if (alph <= 0.) return;
		if (alph < 1.) alph += 1.;
		a1 = alph-1./3.;
		a2 = 1./sqrt(9.*a1);
	}
	 __device__ Doub dev() {
	// Return a gamma deviate by the method of Marsaglia and Tsang.
		Doub u,v,x;
		do {
			do {
				x = Normaldev_BM::dev();
				v = 1. + a2*x;
			} while (v <= 0.);
			v = v*v*v;
			u = doub();
		} while (u > 1. - 0.331*SQR(SQR(x)) &&
			log(u) > 0.5*SQR(x) + a1*(1.-v+log(v))); // Rarely evaluated.
		if (alph == oalph) return a1*v/bet;
		else { // Case where ˛ < 1, per Ripley.
			do u=doub(); while (u == 0.);
			return powf(u,1./oalph)*a1*v/bet;
		}
	}
};


__device__ void d_rnorm(Normaldev_BM* nd,  int n, float mu, float sig, Doub* res)
{
	for(int i=0; i<n; i++) 
		res[i] = nd->dev();
}
	
__device__ void d_rchisq(Gammadev* chi, int n, Doub* res)
{
	for(int i=0; i<n; i++) 
		res[i] = chi->dev();
}



