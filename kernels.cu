/* fast NN operations */

extern "C" {
#include "kernels.h"
#include "cuda.h"
}

#define BLOCK_SIZE 1024

__global__ void memset_kernel(float *p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = v;
    }
}

// return a host pointer to an array of float 1
float *get_ones(int sz) {
    static int size = 0;
    static float *ones = NULL;

    if (sz > size) {
        if (ones) {
            cuda_free(ones);
        }

        cuda_malloc((void **) &ones, sizeof(float) * sz);
        size = sz;
    }

    int grid_size = (sz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    memset_kernel<<<grid_size, BLOCK_SIZE>>>(ones, 1, sz);

    return ones;
}

void linear_create(int in_dim, int out_dim, struct linear *linear) {
    linear->in_dim = in_dim;
    linear->out_dim = out_dim;

    cuda_malloc((void **) &linear->w, sizeof(float) * in_dim * out_dim);
    cuda_malloc((void **) &linear->d, sizeof(float) * in_dim * out_dim);
    cuda_malloc((void **) &linear->m, sizeof(float) * in_dim * out_dim);
    cuda_malloc((void **) &linear->v, sizeof(float) * in_dim * out_dim);
}

/*
Given:
- linear: one linear layer
- input: batch x in_dim
- output: batch x out_dim
Produces:
- output = linear->w x input
We do this under transposition because batch-major memory layout is rumored to be faster
*/
void linear_forward(struct linear *linear, int batch,
        float *input, float *output) {
    int odim = linear->out_dim;
    int idim = linear->in_dim;
    cublas_sgemm(0, 0, odim, batch, idim,
        1, linear->w, odim, input, idim, 0, output, odim);
}

/*
Given:
- linear: one linear layer
- input: batch x in_dim - input during forward pass
- dLdo: batch x out_dim - dL/d(output)
- dLdi: batch x in_dim
Produces:
- dLdi = dL/d(input)
- accumulates dL/dW into layer memory
Note that it is safe for input == dLdi.
*/
void linear_backward(struct linear *linear, int batch,
        float *input, float *dLdo, float *dLdi) {
    int odim = linear->out_dim;
    int idim = linear->in_dim;

    // accumulate: calculate dL/dw
    cublas_sgemm(0, 1, odim, idim, batch,
        1, dLdo, odim, input, idim, 1, linear->d, odim);

    // propagate: calculate dL/d(input)
    cublas_sgemm(1, 0, idim, batch, odim,
        1, linear->w, odim, dLdo, odim, 0, dLdi, idim);
}

__global__ void relu_forward_kernel(int n, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * (input[i] > 0);
}

// input: relu inputs
// output: relu outputs
void relu_forward(int n, float *input, float *output) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_forward_kernel<<<grid_size, BLOCK_SIZE>>>(n, input, output);
}

// pass through gradient when input > 0
__global__ void relu_backward_kernel(int n, float *input, float *grad, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = grad[i] * (input[i] > 0);
}

// input: relu inputs
// grad: gradient w.r.t. input of next layer
// output: gradient w.r.t. input of this layer
void relu_backward(int n, float *input, float *grad, float *output) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_backward_kernel<<<grid_size, BLOCK_SIZE>>>(n, input, grad, output);
}

__global__ void dropout_forward_kernel(int n, float *input, float *dropout, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) output[i] = (dropout[i] < 0.5) * input[i];
}

// input: dropout inputs
// dropout: scratch for storing which entries were dropped out
// output: dropout outputs
void dropout_forward(int n, float *input, float *dropout, float *output) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curand_generate_uniform(dropout, 1);
    dropout_forward_kernel<<<grid_size, BLOCK_SIZE>>>(n, input, dropout, output);
}

// pass through gradient when not dropped out
__global__ void dropout_backward_kernel(int n, float *dropout, float *dLdo, float *dLdi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) {
        dLdi[i] = (dropout[i] < 0.5) * dLdo[i];
    }
}

// dropout: which entries were dropped out
// dLdo: gradient w.r.t. input of next layer
// dLdi: gradient w.r.t. input of this layer
void dropout_backward(int n, float *dropout, float *dLdo, float *dLdi) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dropout_backward_kernel<<<grid_size, BLOCK_SIZE>>>(n, dropout, dLdo, dLdi);
}

void atom_gather_forward(int n_atoms, int n_bonds, int hidden,
        int *a_bonds, int *a2b, float *input, float *output) {
    //cusparse_sgemmi(hidden, n_bonds, n_atoms, n_bonds, 1,
    //    input, n_bonds, get_ones(n_bonds), a_bonds, a2b, 0, output, n_atoms);
    cusparse_scsrmm2(0, 1, n_atoms, hidden, n_bonds, n_bonds,
        1, get_ones(n_bonds), a_bonds, a2b, input, hidden, 0, output, n_atoms);
}

void atom_gather_backward(int n_atoms, int n_bonds, int hidden,
        int *a_bonds, int *a2b, float *input, float *dLdo, float *dLdi) {
    // TODO
}

/*
Given:
- r: number of rows of src1, src2, and dst
- c1: number of columns of src1
- c2: number of columns of src2
- src1: r x c1
- src2: r x c2
- dst: r x (c1 + c2)
Produces:
- dst is a column-wise concatenation of src1 and src2
*/
void concat(int r, int c1, int c2, float *src1, float *src2, float *dst) {
    cuda_memcpy_2d(dst, c1 + c2, src1, c1, c1, r);
    cuda_memcpy_2d(dst + c1, c1 + c2, src2, c2, c2, r);
}

/*
Given:
- r: number of rows of src and dst
- c: number of cols of src
- i1: starting column index for slice
- i2: ending column index for slice
- src: r x c
- dst: r x (i2 - i1)
Produces:
- dst is the slice of columns i1 through i2 out of src
Assume that 0 <= i1 < i2 < c
*/
void slice(int r, int c, int i1, int i2, float *src, float *dst) {
    cuda_memcpy_2d(dst, i2 - i1, src + i1, c, i2 - i1, r);
}

// bond scatter
// TODO optimize with shared memory?
// TODO or just use another appropriate sgemmi?
__global__ void bond_scatter_forward_kernel(int bonds, int hidden,
        int *b2a, int *b2revb, float *a_message, float *messages,
        float *new_messages) {
    int bond = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond < bonds) {
        for (int i = 0; i < hidden; i++) {
            new_messages[bond * hidden + i] =
                a_message[b2a[bond] * hidden + i] -
                messages[b2revb[bond] * hidden + i];
        }
    }
}

void bond_scatter_forward(int bonds, int hidden,
        int *b2a, int *b2revb, float *a_message, float *messages,
        float *new_messages) {
    int grid_size = (bonds + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bond_scatter_forward_kernel<<<grid_size, BLOCK_SIZE>>>(bonds, hidden,
        b2a, b2revb, a_message, messages, new_messages);
}

// two gradients come out of this
void bond_scatter_backward(int bonds, int hidden,
        int *b2a, int *b2revb, float *a_message_in, float *messages_in,
        float *dLdo, float *dLdamesg, float *dLdmesgin) {
    // TODO
}

/*
Given:
- r, c: matrix rows/cols
- input: r x c
- output: r
Produces:
- output[i] = sum_j(input[j][i]) / r
Averages the rows of a matrix by product with all-1 vector.
*/
void average_forward(int r, int c, float *input, float *output) {
    cublas_sgemv(0, c, r, 1, input, c, get_ones(r), 0, output);
}

__global__ void average_backward_kernel(int r, int c, float *dLdo, float *dLdi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < r) {
        for (int j = 0; j < c; j++) {
            dLdi[r * i + j] = dLdo[j];
        }
    }
}

// TODO this is inefficient
void average_backward(int r, int c, float *dLdo, float *dLdi) {
    int grid_size = (r + BLOCK_SIZE - 1) / BLOCK_SIZE;
    average_backward_kernel<<<grid_size, BLOCK_SIZE>>>(r, c, dLdo, dLdi);
}

__global__ void sigmoid_forward_kernel(int n, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1 / (1 + exp(-input[i]));
    }
}

/*
Given:
- n: number of elements
- input: n
- output: n
Produces:
- output[i] = sigmoid(input[i])
*/
void sigmoid_forward(int n, float *input, float *output) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_forward_kernel<<<grid_size, BLOCK_SIZE>>>(n, input, output);
}

__global__ void sigmoid_backward_kernel(int n, float *input, float *dLdo, float *dLdi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float e = exp(-input[i]);
        dLdi[i] = dLdo[i] * e / ((1 + e) * (1 + e));
    }
}

/*
Given:
- n: number of elements
- input: n - input during forward pass
- dLdo: n - dL/d(output)
- dLdi: n
Produces:
- dLdi = dL/d(input)
*/
void sigmoid_backward(int n, float *input, float *dLdo, float *dLdi) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_backward_kernel<<<grid_size, BLOCK_SIZE>>>(n, input, dLdo, dLdi);
}
