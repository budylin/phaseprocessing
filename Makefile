INC = -I../eigen3/
CFL = -Wall -std=c++0x -fopenmp -g -O3 -ffast-math -mavx -msse4.2 -DNDEBUG 

all: convolve_trans

test:
	g++  $(INC) eigen.cpp -o eigen

transp: transptest.cpp
	g++ $(CFL) $(INC) transptest.cpp -o transp

convolve: convolve.cpp
	g++ $(CFL) $(INC) convolve.cpp -o convolve

convolve_trans: convolve_trans.cpp Makefile
	g++ $(CFL) $(INC) convolve_trans.cpp -S -o convolve_trans.S
	g++ $(CFL) $(INC) convolve_trans.cpp -o convolve_trans


fft: convolve_fft.cpp Makefile
	g++ $(CFL) $(INC) convolve_fft.cpp -o convolve_fft
