// $Id: d_rngMars.cu,v 1.6 2010/05/15 14:39:39 afs Exp $
/*
 * "Marsaglia-Multicarry": A _multiply-with-carry_ RNG 
 * following the R implementation
 */ 
 
// Modified for CUDA by A. Ferreira da Silva (C) May 2010


namespace MM_rng {

class __align__(16) Rng {
#define i2_32m1 2.328306437080797e-10/* = 1/(2^32 - 1) */
	unsigned int I1, I2;
public:
	__device__ Rng() : I1(1234), I2(5678) {}
	__device__ Rng(ulint i1, ulint i2) : I1(i1), I2(i2) {
		I1= 36969*(I1 & 0177777) + (I1>>16);
		I2= 18000*(I2 & 0177777) + (I2>>16);
	}
  //   case MARSAGLIA_MULTICARRY:/* 0177777(octal) == 65535(decimal)*/
	__device__ real dev() {
		I1= 36969*(I1 & 0177777) + (I1>>16);
		I2= 18000*(I2 & 0177777) + (I2>>16);
		return fixup(((I1 << 16)^(I2 & 0177777)) * i2_32m1); /* in [0,1) */
	}
	__device__ real fixup(real x)
	{
	    /* ensure 0 and 1 are never returned */
 	   if(x <= 0.0) return 0.5*i2_32m1;
 	   if((1.0 - x) <= 0.0) return 1.0 - 0.5*i2_32m1;
	   return x;
	}
#undef i2_32m1
};

class __align__(16) rngNormal : Rng {
	real BM_norm_keep;
public:
	__device__ rngNormal() : Rng(), BM_norm_keep(0.) {}
	__device__ rngNormal(ulint i1, ulint i2) : Rng(i1, i2), BM_norm_keep(0.) {}
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
	__device__ rngGamma() : rngNormal() {}
	__device__ rngGamma(real alpha, real beta, ulint i1, ulint i2)
		: rngNormal(i1, i2), alpha(alpha), alphain(alpha), beta(beta) {
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
