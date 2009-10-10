CXX=g++ -m64 
AR=ar

# % CXXFLAGS = -O2 -Wall -shared -fPIC

CXXFLAGS = -shared -fPIC  -O2 -Wall 

DIFF = ./sdiff
PRE = ./
MAJOR = 1
MINOR = 0

%.o:           	%.cpp
		$(CXX) $(CXXFLAGS) -c $*.cpp

newmat_lobj = newmat1.o newmat2.o newmat3.o newmat4.o newmat5.o newmat6.o newmat7.o newmat8.o newmatex.o bandmat.o submat.o myexcept.o cholesky.o evalue.o fft.o hholder.o jacobi.o newfft.o sort.o svd.o nm_misc.o newmatrm.o newmat9.o

libnewmat.so:   	$(newmat_lobj)
		$(AR) -cr $@ $(newmat_lobj)
		ranlib $@

newmat1.o:     	newmat1.cpp newmat.h include.h myexcept.h

newmat2.o:     	newmat2.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

newmat3.o:     	newmat3.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

newmat4.o:     	newmat4.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

newmat5.o:     	newmat5.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

newmat6.o:     	newmat6.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

newmat7.o:     	newmat7.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

newmat8.o:     	newmat8.cpp include.h newmat.h newmatrc.h precisio.h myexcept.h controlw.h

newmatex.o:    	newmatex.cpp include.h newmat.h myexcept.h

bandmat.o:     	bandmat.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

submat.o:      	submat.cpp include.h newmat.h newmatrc.h myexcept.h controlw.h

myexcept.o:    	myexcept.cpp include.h myexcept.h

cholesky.o:    	cholesky.cpp include.h newmat.h newmatrm.h myexcept.h

evalue.o:      	evalue.cpp include.h newmatap.h newmatrm.h precisio.h newmat.h myexcept.h

fft.o:         	fft.cpp include.h newmatap.h newmat.h myexcept.h

hholder.o:     	hholder.cpp include.h newmatap.h newmat.h myexcept.h

jacobi.o:      	jacobi.cpp include.h newmatap.h precisio.h newmatrm.h newmat.h myexcept.h

newfft.o:      	newfft.cpp newmatap.h newmat.h include.h myexcept.h

sort.o:        	sort.cpp include.h newmatap.h newmat.h myexcept.h

svd.o:         	svd.cpp include.h newmatap.h newmatrm.h precisio.h newmat.h myexcept.h

nm_misc.o:     	nm_misc.cpp include.h newmatap.h newmat.h myexcept.h

newmatrm.o:    	newmatrm.cpp newmat.h newmatrm.h include.h myexcept.h

newmat9.o:     	newmat9.cpp include.h newmat.h newmatio.h newmatrc.h myexcept.h controlw.h


all: $(libnewmat.so) 

clean:
	@-rm -f *.o *.so 

.PHONY: all clean
