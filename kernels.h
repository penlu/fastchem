// constant bias
struct bias {
    int dim;
    float *b; // constants
    float *d;
    float *m;
    float *v;
};

// matrix multiplication
struct linear {
    int in_dim;
    int out_dim;
    float *w; // weights
    float *d; // gradients
    float *m; // 1st moment vector (Adam)
    float *v; // 2nd moment vector (Adam)
};

float *get_ones(int);

void bias_create(int, struct bias *);
void bias_init(struct bias *);
void bias_forward(struct bias *, int, float *);
void bias_backward(struct bias *, int, float *);
void bias_adam(struct bias *, int, float, float, float);

void linear_create(int, int, struct linear *);
void linear_init(struct linear *);
void linear_forward(struct linear *, int, float *, float *);
void linear_backward(struct linear *, int, float *, float *, float *);
void linear_adam(struct linear *, int, float, float, float);

void relu_forward(int, float *, float *);
void relu_backward(int, float *, float *, float *);

void dropout_forward(int, float *, float *, float *);
void dropout_backward(int, float *, float *, float *);

void atom_gather_forward(int, int, int, int *, int *, float *, float *);
void atom_gather_backward(int, int, int, int *, int *, float *, float *);

void concat(int, int, int, float *, float *, float *);
void slice(int, int, int, int, float *, float *);

void bond_scatter_forward(int, int, int *, int *, float *, float *, float *);
void bond_scatter_backward(int, int, int, int *, int *, int *, float *, float *, float *);

void average_forward(int, int, float *, float *);
void average_backward(int, int, float *, float *);

void sigmoid_forward(int, float *, float *);
void sigmoid_backward(int, float *, float *, float *);

void bceloss_forward(int, float *, float *,float *);
void bceloss_backward(int, float *, float *, float *);
