CC=gcc
NVCC=nvcc
CFLAGS=-fopenmp -O3 -Wextra -std=c11 -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lstdc++ -g
CUDAFLAGS=-std=c++11 -c -arch=sm_61
OBJ=matrixMul.o

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

matrixMul: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o matrixMul
