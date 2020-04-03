CC=gcc
NVCC=nvcc
CFLAGS=-fopenmp -O3 -Wextra -std=c11 -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lcurand -lstdc++ -g -DDEBUG
CUDAFLAGS=-std=c++11 -c -arch=sm_61 -g
OBJ=main.o mol.o datapt.o mpn.o kernels.o cuda.o

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
