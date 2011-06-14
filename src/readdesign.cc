#include <iostream>
#include <vector>
#include <iterator>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

using namespace std;

extern "C" {
char *modelkey(FILE *fd, char *keyline, const char *key);
vector<double> readdesign(char* filename, int *nevs, int *npts);
void readmodel(char** filename, int* nevs, int* npts, double* dm);
}

char *modelkey(FILE *fd, char *keyline, const char *key)
{
  while (fgets(keyline, 1000, fd)!=NULL)
    if ( (strncmp(keyline,key,strlen(key))==0) )
      return keyline+strlen(key);
  printf("Error: key \"%s\" not found.\n",key);
  exit(1);
}

vector<double> readdesign(char* filename, int *nevs, int *npts)
{
  FILE *designFile;
  char keyline[1010];
  vector<double> model;
  if((designFile=fopen(filename,"r"))==NULL) {
      *npts=0;
      return model;
  }
  *nevs=(int)atof(modelkey(designFile,keyline,"/NumWaves"));
  *npts=(int)atof(modelkey(designFile,keyline,"/NumPoints"));
  atof(modelkey(designFile,keyline,"/Matrix"));
  model.resize( *nevs * *npts );
  for(int i=0; i<*nevs * *npts; i++)
    fscanf(designFile,"%lf",&model[i]);
  return model;
}

void readmodel(char** filename, int* nevs, int* npts, double* dm)
{
  vector<double> v=readdesign(filename[0], nevs, npts);
	vector<double>::iterator vitb = v.begin();
	vector<double>::iterator vite = v.end();
	while(vitb != vite) *dm++ = *vitb++;
}

