
__device__ void choldcU(float* a, int* pn, float* y)  
{
	int n =  *pn;
	int i,j,k;
	float sum;
	unsigned int ij, ik, jk, ii, ji;
	//
	for (i=0;i<n;i++) 
		for (j=0;j<n;j++) {
			ij=i+n*j;
			y[ij] = a[ij]; 
		}
	//
	for (i=0;i<n;i++) {
		for (j=i;j<n;j++) {
			ij=i+n*j;
			for (sum=y[ij],k=i-1;k>=0;k--) {
		 		ik=i+n*k; jk=j+n*k;
				sum -= y[ik]*y[jk];
			}
			ii=i+n*i;
			if (i == j) {
				if (sum <= float(0.0))  //	A, with rounding errors, is not positive-deﬁnite.
					return;
					// throw("\n*** Cholesky failed\n")
				y[ii]=sqrt(sum);
			}
			else {
				ji=j+n*i;
				y[ji]=sum/y[ii];
			}
		}
	}
	for (i=0;i<n;i++) // output as Upper
		for (j=0;j<i;j++) {
			ji=j+n*i;
			ij=i+n*j;
			y[ji] = y[ij];
			y[ij] = float(0.0);
		}

/*
	int l;
	for (i=0;i<n;i++){ // by columns
		for (j=0;j<n;j++) {
			l=i+n*j;
			printf("%e ", y[l]);
		}
		printf("\n");
	}
	printf("----------\n");
*/
}

//----------------------------------------------------------------------
// Input as L^T
__device__ void elsolveU(float* el, int* b, float* y, int* pn)
{
// Solve L y = b, where L is the lower triangular matrix in the stored Cholesky decomposi-
// tion. b[0..n-1] is input as the right-hand side vector. The solution vector is returned in
// y[0..n-1].
	int n = *pn;
	int i,k;
	int ki, ii;
	float sum;
	// if (b.size() != n || y.size() != n) throw("bad lengths");
	for (i=n-1;i>=0;i--) { // Solve L^T x = y.
		for (sum=b[i],k=i+1;k<n;k++){
			ki=k*n+i;
			sum -= el[ki]*y[k];
		}
		ii=i*n+i;
		y[i]=sum/el[ii];
	}
}


//----------------------------------------------------------------------

__device__ void mdgbacksolve(float* a, int* pn, float* y)
{
	int n =  *pn;
	// __shared__ float el[25];
	float el[25];
  choldcU(a,pn,el);
	//
	// __shared__ int b[5]; 
	int b[5]; 

	for(int i=0; i < n; i++) b[i]= 0;
	for(int i=0; i < n; i++) {
		b[i]=1;
		elsolveU(el, b , y+i*n, pn); 
		b[i]=0;
	}

}

//----------------------------------------------------------------------

__device__ void mtcrossp(float *a, float *b, float *c, int *pn)
{
	int n = *pn;
	float sum;
	int ik, kj, ij;
	//
  for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
		  sum = 0.0;
		  for (int k=0; k<n; k++) {
				ik=i+n*k;
				kj=k*n+j;
				sum += a[ik] * b[kj];
			}
			ij=i+n*j;
			c[ij] = sum;
		}
}

//----------------------------------------------------------------------
// product of matrix (mxn) x vector (n) 

__device__ void mvprod(float *a, float *b, float *c, int *pm, int *pn)
{
	int m = *pm;
	int n = *pn;
	float sum;
	int ik;
	//
  for (int i=0; i<m; i++) {
		sum = 0.0;
		for (int k=0; k<n; k++) {
			ik=i+m*k;
			sum += a[ik] * b[k];
		}
		c[i] = sum;
	}
}

//----------------------------------------------------------------------
/*
// product of matrix (mxn) x vector (m) 

__device__ void mvprodm(float *a, float *b, float *c, int *pm)
{
	int m = *pm;
	float sum;
	int ik;
	//
  for (int i=0; i<m; i++) {
		sum = 0.0;
		for (int k=0; k<m; k++) {
			ik=i+m*k;
			sum += a[ik] * b[k];
		}
		c[i] = sum;
	}
}

*/

//----------------------------------------------------------------------
//
// product of vectors 1xm X mx1
// 
__device__ void vprod(float *a, float *b, float *c, int *pm)
{
	int m = *pm;
	float sum;
	//
	sum = 0.0;
  for (int i=0; i<m; i++) {
		sum += a[i] * b[i];
	}
  *c = sum;
}

//----------------------------------------------------------------------


// t(matrix)*vector - with R-order (by column)
__device__ void mvtcrossp(float *a, float *b, float *c, int *pm, int* pn)
{
	int m = *pm;
	int n = *pn;
	float sum;
	int ik;
	//
  for (int i=0; i<n; i++) {
		sum = 0.0;
		for (int k=0; k<m; k++) {
			ik=m*i+k;
			sum += a[ik] * b[k];
		}
		c[i] = sum;
	}
}


//----------------------------------------------------------------------
