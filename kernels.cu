/* fast NN operations */

extern "C" {
#include "kernels.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
}

#define BLOCK_SIZE 1024

__global__ void memset_kernel(float *p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = v;
    }
}

// return a device pointer to an array of float 1
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

void linear_init(struct linear *linear) {
    curand_generate_normal(linear->w, linear->in_dim * linear->out_dim, 0, 1);
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

#define EPSILON 0.00000001
__global__ void linear_adam_kernel(int n, int step,
        float alpha, float beta1, float beta2,
        float *w, float *d, float *m, float *v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float mhat = m[i] / (1 - pow(beta1, step));
        float vhat = v[i] / (1 - pow(beta2, step));

        w[i] = w[i] - alpha * mhat / (sqrt(vhat) + EPSILON);
    }
}

__global__ void elementwise_square(int n, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * x[i];
    }
}

void linear_adam(struct linear *linear, int step, float alpha, float beta1, float beta2) {
    int n = linear->in_dim * linear->out_dim;
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cublas_sscal(n, beta1, linear->m, 1);
    cublas_saxpy(n, 1 - beta1, linear->d, 1, linear->m, 1);

    elementwise_square<<<grid_size, BLOCK_SIZE>>>(n, linear->d);

    cublas_sscal(n, beta2, linear->v, 1);
    cublas_saxpy(n, 1 - beta2, linear->d, 1, linear->v, 1);

    linear_adam_kernel<<<grid_size, BLOCK_SIZE>>>(n, step,
        alpha, beta1, beta2, linear->w, linear->d, linear->m, linear->v);

    memset_kernel<<<grid_size, BLOCK_SIZE>>>(linear->d, 0, n);
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

// gather bond messages onto atoms
void atom_gather_forward(int n_atoms, int n_bonds, int hidden,
        int *a_bonds, int *a2b, float *input, float *output) {
    cusparse_sgemmi(hidden, n_atoms, n_bonds, n_bonds, 1,
        input, hidden, get_ones(n_bonds), a_bonds, a2b, 0, output, hidden);
}

void atom_gather_backward(int n_atoms, int n_bonds, int hidden,
        int *a_bonds, int *a2b, float *dLdo, float *dLdi) {
    int *a_bondsT;
    int *a2bT;
    cuda_malloc((void **) &a_bondsT, sizeof(int) * (n_bonds + 1));
    cuda_malloc((void **) &a2bT, sizeof(int) * n_bonds);

    cusparse_scsr2csc(n_atoms, n_bonds, n_bonds,
        get_ones(n_bonds), a_bonds, a2b,
        get_ones(n_bonds), a2bT, a_bondsT);
    cusparse_sgemmi(hidden, n_bonds, n_atoms, n_bonds, 1,
        dLdo, hidden, get_ones(n_bonds), a_bondsT, a2bT, 0, dLdi, hidden);

    cuda_free(a_bondsT);
    cuda_free(a2bT);
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
    int dst_width = (c1 + c2) * sizeof(float);
    cuda_memcpy_2d(dst, dst_width,
        src1, c1 * sizeof(float), c1 * sizeof(float), r);
    cuda_memcpy_2d(dst + c1, dst_width,
        src2, c2 * sizeof(float), c2 * sizeof(float), r);
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
    int width = (i2 - i1) * sizeof(float);
    cuda_memcpy_2d(dst, width,
        src + i1, c * sizeof(float), width, r);
}

// scatter atom messages onto bonds
// TODO optimize with shared memory? or find an appropriate gemmi?
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

void bond_scatter_forward(int n_bonds, int hidden,
        int *b2a, int *b2revb, float *a_message, float *messages,
        float *new_messages) {
    int grid_size = (n_bonds + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bond_scatter_forward_kernel<<<grid_size, BLOCK_SIZE>>>(n_bonds, hidden,
        b2a, b2revb, a_message, messages, new_messages);
}

__global__ void bond_scatter_backward_amesgkernel(int n_bonds,
        int *b2revb, int *a2b, int *a2brev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_bonds) {
        a2brev[i] = b2revb[a2b[i]];
    }
}

__global__ void bond_scatter_backward_mesgkernel(int n_bonds, int hidden,
        int *b2revb, float *dLdo, float *dLdmesg) {
    int bond = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond < n_bonds) {
        for (int i = 0; i < hidden; i++) {
            dLdmesg[b2revb[bond] * hidden + i] = -dLdo[bond * hidden + i];
        }
    }
}

// two gradients come out of this
void bond_scatter_backward(int n_atoms, int n_bonds, int hidden,
        int *a_bonds, int *a2b, int *b2revb,
        float *dLdo, float *dLdamesg, float *dLdmesg) {
    int grid_size = (n_bonds + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *a2brev;
    cuda_malloc((void **) &a2brev, sizeof(float) * n_bonds);
    bond_scatter_backward_amesgkernel<<<grid_size, BLOCK_SIZE>>>(n_bonds,
        b2revb, a2b, a2brev);

    // handle a_message gradients
    cusparse_sgemmi(hidden, n_atoms, n_bonds, n_bonds, 1,
        dLdo, hidden, get_ones(n_bonds), a_bonds, a2brev, 0, dLdamesg, hidden);

    // handle message gradients
    bond_scatter_backward_mesgkernel<<<grid_size, BLOCK_SIZE>>>
        (n_bonds, hidden, b2revb, dLdo, dLdmesg);

    cuda_free(a2brev);
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
    cublas_sgemv(0, c, r, 1./r, input, c, get_ones(r), 0, output);
}

__global__ void average_backward_kernel(int r, int c, float *dLdo, float *dLdi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < r) {
        for (int j = 0; j < c; j++) {
            dLdi[c * i + j] = dLdo[j] / r;
        }
    }
}

// TODO this is inefficient
// TODO use cublasSger instead for efficiency?
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
        float x = 1 / (1 + exp(-input[i]));
        dLdi[i] = dLdo[i] * x * (1 - x);
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

__global__ void bceloss_forward_kernel(int n, float *target, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (input[i] > 1) {
            output[i] = input[i] + log1p(exp(-input[i])) - target[i] * input[i];
        } else {
            output[i] = log1p(exp(input[i])) - target[i] * input[i];
        }
    }
}

void bceloss_forward(int n, float *target, float *input, float *output) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bceloss_forward_kernel<<<grid_size, BLOCK_SIZE>>>(n, target, input, output);
}

__global__ void bceloss_backward_kernel(int n, float *target, float *input, float *dLdi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dLdi[i] = 1/(1 + exp(-input[i])) - target[i];
    }
}

void bceloss_backward(int n, float *target, float *input, float *dLdi) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bceloss_backward_kernel<<<grid_size, BLOCK_SIZE>>>(n, target, input, dLdi);
}
