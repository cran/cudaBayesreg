CXX=g++ -m64

AR=ar
MAKE=make

CPPFLAGS = -shared -fPIC  -O2 -Wall -I/usr/local/cuda//include -I.

%.o:           	%.cpp
		$(CXX) $(CPPFLAGS) -c $*.cpp

CUTILFILES := 	bank_checker.o    \
		cmd_arg_reader.o  \
		cutil.o           \
		stopwatch.o       \
		stopwatch_linux.o \
		multithreading.o

libcutil.so: $(CUTILFILES)
		$(AR) -cr $@ $(CUTILFILES)
		ranlib $@

all: $(libcutil.so) 

clean:
	@-rm -f *.o *.so 

.PHONY: all clean
