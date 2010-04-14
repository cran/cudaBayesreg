
#define Ullong unsigned long long int
#define Ulong unsigned long int

// #define Doub double
#define Doub float

#define Uint unsigned int
#define Int int

// #define MAX(x,y) ((x)>(y)?(x):(y))
// #define MIN(x,y) ((x)<(y)?(x):(y))

// SQR(a) Square a float value.
// DSQR(a) Square a double value.
__device__ static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
__device__ static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)


#define UINT32_MAX             (4294967295U)
#define UINT64_MAX             (__UINT64_C(18446744073709551615))

