CXX=g++
AR=ar
MAKE=make

ifeq (TRUE,TRUE)
  NVCC=/usr/local/cuda//bin/nvcc
  NVCCFLAGS = --compiler-options -fno-strict-aliasing --compiler-options -DNDEBUG=1 -O3 -Xcompiler "-fPIC" -Xlinker "-fPIC" -I/usr/local/cuda//include -I./newmat11 -I/usr/local/lib64/R/include
  NVCCLINKS = -shared -Xcompiler "-fPIC" -Xlinker "-fPIC" -lRmath -lcuda -L./newmat11 -lnewmat
endif

CPPFLAGS = -shared -fPIC -DNDEBUG=1 -O3 -pipe -Wall -I/usr/local/cuda//include -I./newmat11 -I/usr/local/lib64/R/include
CPPFLAGS2 = -shared -fPIC -DNDEBUG=1 -O3 -pipe -Wall -I/usr/local/lib64/R/include

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

#TARGET : libnewmat.so cudaBayesreg.so 
TARGET : readdesign.so libnewmat.so cudaBayesreg.so 

cuda_obj = cudaMultireg.o 

utils_obj = utilsRmultireg.o utilsNewmat.o 

readdesign_obj = readdesign.o 

readdesign.so: $(readdesign_obj)

#	$(CXX) $(readdesign_obj) $(CPPFLAGS2) -o readdesign.so

libnewmat.so:
	( cd newmat11 ; $(MAKE) )

cudaBayesreg.so: $(cuda_obj) $(utils_obj) $(readdesign_obj) 
	$(NVCC) $(cuda_obj) $(utils_obj) $(readdesign_obj) $(NVCCLINKS) -o cudaBayesreg.so

all: $(TARGET) 

clean:
	( rm -f *.o *.so ) 
	( cd newmat11 ; $(MAKE) clean )

.PHONY: all clean
