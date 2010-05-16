//$Id: utilsSeeds.cc,v 1.1 2010/05/15 15:55:57 afs Exp $
//
#include <climits>
#include <cstdio>
#include <ctime>

// typedef unsigned long ulint; 

ulint hashseed( time_t t, clock_t c )
{
	static ulint differ = 0;  
	ulint h1 = 0;
	unsigned char *p = (unsigned char *) &t;
	for( size_t i = 0; i < sizeof(t); ++i )
	{
		h1 *= UCHAR_MAX + 2U;
		h1 += p[i];
	}
	ulint h2 = 0;
	p = (unsigned char *) &c;
	for( size_t j = 0; j < sizeof(c); ++j )
	{
		h2 *= UCHAR_MAX + 2U;
		h2 += p[j];
	}
	return ( h1 + differ++ ) ^ h2;
}

ulint rseed()
{
	FILE* urandom = fopen( "/dev/urandom", "rb" );
	if( urandom )
	{
		int N = 1;
		ulint bigSeed;
		register ulint *s = &bigSeed;
		register int i = N;
		register bool success = true;
		while( success && i-- )
		{
			success = fread( s, sizeof(ulint), 1, urandom );
			*s++ &= 0xffffffff;  // filter in case ulint > 32 bits
		}
		fclose(urandom);
		if( success ) { // seed( bigSeed, N );
			return bigSeed; }
	}
	return hashseed( time(NULL), clock() );
}

 
