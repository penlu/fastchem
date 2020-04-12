CC=gcc
NVCC=nvcc
INCLUDE=-I. -I/usr/local/cuda/include
LIBS=-L/usr/local/cuda/lib64 -lm -lcudart -lcublas -lcusparse -lcurand -lstdc++

CFLAGS=-fopenmp -O3 -Wextra -std=c11 -g $(INCLUDE) $(LIBS) #-DDEBUG
CUDAFLAGS=-std=c++11 -c -arch=sm_61 -g
OBJ=main.o mol.o datapt.o batch.o mpn.o kernels.o cuda.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

all: main test

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

test: test.o mol.o datapt.o cuda.o kernels.o
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o main test
