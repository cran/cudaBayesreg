CXX=g++
AR=ar
MAKE=make

ifeq (TRUE,TRUE)
  NVCC=/usr/local/cuda//bin/nvcc
  NVCCFLAGS = --compiler-bindir /tmp --compiler-options -fno-strict-aliasing -Xcompiler "-fPIC" -Xlinker "-fPIC" -I/usr/local/cuda//include -I./newmat11 -I/usr/include/R 
  NVCCLINKS = -shared -Xcompiler "-fPIC" -Xlinker "-fPIC" -L/usr/local/lib64/R/lib -lR -L/usr/local/lib64/R -lRmath -lcuda -L./newmat11 -lnewmat 
endif

CPPFLAGS = -shared -fPIC  -O2 -pipe -Wall -I/usr/local/cuda//include -I./newmat11 -I/usr/include/R

DIFF = ./sdiff
PRE = ./
MAJOR = 1
MINOR = 0

%.o:           	%.cu
		$(NVCC) $(NVCCFLAGS) -c $*.cu
%.o:           	%.cc
		$(CXX) $(CPPFLAGS) -c $*.cc
%.o:           	%.cpp
		$(CXX) $(CPPFLAGS) -c $*.cpp

TARGET : libnewmat.so cudaBayesreg.so

cuda_obj = cudaMultireg.o 

utils_obj = utilsRmultireg.o utilsNewmat.o

libnewmat.so:
	( cd newmat11 ; $(MAKE) )

cudaBayesreg.so: $(cuda_obj) $(utils_obj) 
	$(NVCC) $(cuda_obj) $(utils_obj) $(NVCCLINKS) -o cudaBayesreg.so

all: $(TARGET) 

clean:
	( rm -f *.o *.so ) 
	( cd newmat11 ; $(MAKE) clean )

.PHONY: all clean
