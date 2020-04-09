#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mol.h"
#include "datapt.h"
#include "cuda.h"
#include "kernels.h"

#define MP_DEPTH 3
#define FC_DEPTH 2
#define HIDDEN 10

// storing intermediates for a layer
struct act {
    float *linear_act;  // outputs of linear
    float *dropout_act; // dropout activations
    float *output;      // final output
};

struct act *act_create(int size) {
    struct act *act = malloc(sizeof(struct act));
    cuda_malloc((void **) &act->linear_act, sizeof(float) * size);
    cuda_malloc((void **) &act->dropout_act, sizeof(float) * size);
    cuda_malloc((void **) &act->output, sizeof(float) * size);
    return act;
}

void act_free(struct act *act) {
    cuda_free(act->linear_act);
    cuda_free(act->dropout_act);
    cuda_free(act->output);
    free(act);
}

// each "layer" is a matrix multiply followed by a ReLU and dropout
void layer_forward(struct linear *linear, int batch, float *input, struct act *act) {
    linear_forward(linear, batch, input, act->linear_act);
    relu_forward(batch * linear->out_dim, act->linear_act, act->output);
    dropout_forward(batch * linear->out_dim, act->output, act->dropout_act, act->output);
}

// note that it is safe for input == dLdi
// expect dLdo in act->output
void layer_backward(struct linear *linear, int batch,
        float *input, struct act *act, float *dLdi) {
    dropout_backward(batch * linear->out_dim, act->dropout_act,
        act->output, act->output);
    relu_backward(batch * linear->out_dim, act->linear_act,
        act->output, act->output);
    linear_backward(linear, batch, input, act->output, dLdi);
}

struct mpn {
    // network weights
    struct linear W_i; // bond_fdim x hidden_size
    struct linear W_h; // hidden_size x hidden_size
    struct linear W_o; // (atom_fdim + hidden_size) x hidden_size
    struct linear fc[FC_DEPTH]; // hidden_size x hidden_size
    struct linear fco; // hidden_size x 1

    // activation storage
    // TODO surely there is a better way
    // TODO adopt more sustainable layer-oriented system

    // for the device-side mol
    struct mol d_mol;

    // for message passing
    struct act *mp_acts[MP_DEPTH + 1];  // message-passing activation storage
    float *mp_atoms[MP_DEPTH];          // outputs of atom-gather
    float *mp_bonds[MP_DEPTH];          // outputs of bond-scatter

    // for molecule embedding
    float *out_atoms;
    float *out_atoms_f;
    struct act *out_act;
    float *embedding;

    // for final fully-connected layers
    struct act *fc_acts[FC_DEPTH];

    // for output
    float *fco_act;
    float fco_out;
};

void mol_to_device(struct mol *mol, struct mol *d_mol) {
    // copy molecule to device
    cuda_malloc((void **) &d_mol->f_atoms, sizeof(float) * mol->n_atoms * ATOM_FDIM);
    cuda_malloc((void **) &d_mol->f_bonds, sizeof(float) * mol->n_bonds * BOND_FDIM);
    cuda_malloc((void **) &d_mol->a_bonds, sizeof(int) * (mol->n_atoms + 1));
    cuda_malloc((void **) &d_mol->a2b, sizeof(int) * mol->n_bonds);
    cuda_malloc((void **) &d_mol->b2a, sizeof(int) * mol->n_bonds);
    cuda_malloc((void **) &d_mol->b2revb, sizeof(int) * mol->n_bonds);

    cuda_memcpy_htod(d_mol->f_atoms, mol->f_atoms, sizeof(float) * mol->n_atoms * ATOM_FDIM);
    cuda_memcpy_htod(d_mol->f_bonds, mol->f_bonds, sizeof(float) * mol->n_bonds * BOND_FDIM);
    cuda_memcpy_htod(d_mol->a_bonds, mol->a_bonds, sizeof(int) * (mol->n_atoms + 1));
    cuda_memcpy_htod(d_mol->a2b, mol->a2b, sizeof(int) * mol->n_bonds);
    cuda_memcpy_htod(d_mol->b2a, mol->b2a, sizeof(int) * mol->n_bonds);
    cuda_memcpy_htod(d_mol->b2revb, mol->b2revb, sizeof(int) * mol->n_bonds);
}

void free_dmol(struct mol *d_mol) {
    cuda_free(d_mol->f_atoms);
    cuda_free(d_mol->f_bonds);
    cuda_free(d_mol->a_bonds);
    cuda_free(d_mol->a2b);
    cuda_free(d_mol->b2a);
    cuda_free(d_mol->b2revb);
}

struct mpn *mpn_create() {
    struct mpn *mpn = malloc(sizeof(struct mpn));

    linear_create(BOND_FDIM, HIDDEN, &mpn->W_i);
    linear_create(HIDDEN, HIDDEN, &mpn->W_h);
    linear_create(ATOM_FDIM + HIDDEN, HIDDEN, &mpn->W_o);
    for (int i = 0; i < FC_DEPTH; i++) {
        linear_create(HIDDEN, HIDDEN, &mpn->fc[i]);
    }
    // TODO this is a vector dot and should not use sgemm
    linear_create(HIDDEN, 1, &mpn->fco);

    return mpn;
}

void mpn_init(struct mpn *mpn) {
    linear_init(&mpn->W_i);
    linear_init(&mpn->W_h);
    linear_init(&mpn->W_o);
    for (int i = 0; i < FC_DEPTH; i++) {
        linear_init(&mpn->fc[i]);
    }
    linear_init(&mpn->fco);
}

// allocate memory for intermediate activations
void alloc_intermediate(struct mpn *mpn, struct mol *mol) {
    // move input to GPU
    mol_to_device(mol, &mpn->d_mol);

    for (int i = 0; i < MP_DEPTH + 1; i++) {
        mpn->mp_acts[i] = act_create(mol->n_bonds * HIDDEN);
    }
    for (int i = 0; i < MP_DEPTH; i++) {
        cuda_malloc((void **) &mpn->mp_atoms[i], sizeof(float) * mol->n_atoms * HIDDEN);
        cuda_malloc((void **) &mpn->mp_bonds[i], sizeof(float) * mol->n_bonds * HIDDEN);
    }

    cuda_malloc((void **) &mpn->out_atoms, sizeof(float) * mol->n_atoms * HIDDEN);
    cuda_malloc((void **) &mpn->out_atoms_f, sizeof(float) * mol->n_atoms * (ATOM_FDIM + HIDDEN));
    cuda_malloc((void **) &mpn->embedding, sizeof(float) * HIDDEN);
    mpn->out_act = act_create(mol->n_atoms * HIDDEN);

    for (int i = 0; i < FC_DEPTH; i++) {
        mpn->fc_acts[i] = act_create(mol->n_bonds * HIDDEN);
    }

    cuda_malloc((void **) &mpn->fco_act, sizeof(float));
}

void free_intermediate(struct mpn *mpn) {
    free_dmol(&mpn->d_mol);

    for (int i = 0; i < MP_DEPTH + 1; i++) {
        act_free(mpn->mp_acts[i]);
    }
    for (int i = 0; i < MP_DEPTH; i++) {
        cuda_free(mpn->mp_atoms[i]);
        cuda_free(mpn->mp_bonds[i]);
    }

    cuda_free(mpn->out_atoms);
    cuda_free(mpn->out_atoms_f);
    cuda_free(mpn->embedding);
    act_free(mpn->out_act);

    for (int i = 0; i < FC_DEPTH; i++) {
        act_free(mpn->fc_acts[i]);
    }

    cuda_free(mpn->fco_act);
}

void print_matrix(int rows, int cols, float *d_mat) {
    float *mat = (float *) malloc(sizeof(float) * rows * cols);
    cuda_memcpy_dtoh(mat, d_mat, sizeof(float) * rows * cols);
    for (int j = 0; j < rows; j++) {
        printf("%d: ", j);
        for (int i = 0; i < cols; i++) {
            printf("%4.2f ", mat[i + cols * j]);
        }
        printf("\n");
    }
    printf("\n");
    free(mat);
}

float bceloss(float target, float z) {
    if (z > 1) {
        return z + log1p(exp(-z)) - target * z;
    } else {
        return log1p(exp(z)) - target * z;
    }
}

float bceloss_grad(float target, float z) {
    return 1/(1 + exp(-z)) - target;
}

// message-passing network
float mpn_forward(struct mpn *mpn, struct mol *mol, float target) {
    int n_bonds = mol->n_bonds;
    int n_atoms = mol->n_atoms;

    alloc_intermediate(mpn, mol);

    // messages = ReLU(mpn.W_i(mol.f_bonds))
    layer_forward(&mpn->W_i, n_bonds, mpn->d_mol.f_bonds, mpn->mp_acts[0]);

    // message-passing
    for (int i = 0; i < MP_DEPTH; i++) {
        // for each atom:
        //     a_message[atom] = sum incoming messages from neighbor atoms
        // for each bond:
        //     new_messages = a_message[b2a[bond]] - messages[b2rev[bond]]
        // messages = dropout(ReLU(mpn.W_h(new_messages)))

        atom_gather_forward(n_atoms, n_bonds, HIDDEN, mpn->d_mol.a_bonds, mpn->d_mol.a2b,
            mpn->mp_acts[i]->output, mpn->mp_atoms[i]);
        bond_scatter_forward(n_bonds, HIDDEN, mpn->d_mol.b2a, mpn->d_mol.b2revb,
            mpn->mp_atoms[i], mpn->mp_acts[i]->output, mpn->mp_bonds[i]);
        layer_forward(&mpn->W_h, n_bonds, mpn->mp_bonds[i], mpn->mp_acts[i + 1]);
    }

    // calculate a_messages one more time
    atom_gather_forward(n_atoms, n_bonds, HIDDEN,
        mpn->d_mol.a_bonds, mpn->d_mol.a2b,
        mpn->mp_acts[MP_DEPTH]->output, mpn->out_atoms);

    // concatenate f_atoms to each one
    concat(n_atoms, ATOM_FDIM, HIDDEN, mpn->d_mol.f_atoms,
        mpn->out_atoms, mpn->out_atoms_f);

    // dropout(ReLU(mpn.W_o(that)))
    layer_forward(&mpn->W_o, n_atoms, mpn->out_atoms_f, mpn->out_act);

    // average these to get the molecule embedding
    average_forward(n_atoms, HIDDEN, mpn->out_act->output, mpn->embedding);

    // one fully-connected feed-forward network, three layers
    // linear o activation o dropout
    // linear o activation to final value
    layer_forward(&mpn->fc[0], 1, mpn->embedding, mpn->fc_acts[0]);
    for (int i = 1; i < FC_DEPTH; i++) {
        layer_forward(&mpn->fc[i], 1, mpn->fc_acts[i - 1]->output, mpn->fc_acts[i]);
    }

    // TODO this should be a dot product
    linear_forward(&mpn->fco, 1, mpn->fc_acts[FC_DEPTH - 1]->output, mpn->fco_act);

    // get the goods!!!
    cuda_memcpy_dtoh(&mpn->fco_out, mpn->fco_act, sizeof(float));

    return bceloss(target, mpn->fco_out);
}

float mpn_backward(struct mpn *mpn, struct mol *mol, float target) {
    int n_bonds = mol->n_bonds;
    int n_atoms = mol->n_atoms;

    float grad = bceloss_grad(target, mpn->fco_out);
    cuda_memcpy_htod(mpn->fco_act, &grad, sizeof(float));

    linear_backward(&mpn->fco, 1, mpn->fc_acts[FC_DEPTH - 1]->output,
        mpn->fco_act, mpn->fc_acts[FC_DEPTH - 1]->output);

    for (int i = FC_DEPTH - 1; i > 0; i--) {
        layer_backward(&mpn->fc[i], 1, mpn->fc_acts[i - 1]->output,
            mpn->fc_acts[i], mpn->fc_acts[i - 1]->output);
    }
    layer_backward(&mpn->fc[0], 1, mpn->embedding, mpn->fc_acts[0], mpn->embedding);

    average_backward(n_atoms, HIDDEN, mpn->embedding, mpn->out_act->output);

    layer_backward(&mpn->W_o, n_atoms, mpn->out_atoms_f,
        mpn->out_act, mpn->out_atoms_f);

    slice(n_atoms, ATOM_FDIM + HIDDEN, ATOM_FDIM, ATOM_FDIM + HIDDEN,
        mpn->out_atoms_f, mpn->out_atoms);

    atom_gather_backward(n_atoms, n_bonds, HIDDEN, mpn->d_mol.a_bonds, mpn->d_mol.a2b,
        mpn->out_atoms, mpn->mp_acts[MP_DEPTH]->output);

    // TODO can we avoid this allocation?
    float *dLdmesg;
    cuda_malloc((void **) &dLdmesg, sizeof(float) * n_bonds * HIDDEN);
    for (int i = MP_DEPTH - 1; i >= 0; i--) {
        layer_backward(&mpn->W_h, n_bonds, mpn->mp_bonds[i],
            mpn->mp_acts[i + 1], mpn->mp_bonds[i]);
        bond_scatter_backward(n_atoms, n_bonds, HIDDEN,
            mpn->d_mol.a_bonds, mpn->d_mol.a2b, mpn->d_mol.b2revb,
            mpn->mp_bonds[i], mpn->mp_atoms[i], dLdmesg);
        atom_gather_backward(n_atoms, n_bonds, HIDDEN,
            mpn->d_mol.a_bonds, mpn->d_mol.a2b,
            mpn->mp_atoms[i], mpn->mp_acts[i]->output);
        cublas_saxpy(n_bonds * HIDDEN, 1, dLdmesg, 1, mpn->mp_acts[i]->output, 1);
    }
    cuda_free(dLdmesg);

    // don't trash the input
    // TODO don't waste time here
    float *garbage;
    cuda_malloc((void **) &garbage, sizeof(float) * n_bonds * BOND_FDIM);
    layer_backward(&mpn->W_i, n_bonds, mpn->d_mol.f_bonds, mpn->mp_acts[0], garbage);
    cuda_free(garbage);

    free_intermediate(mpn);
}

void mpn_adam(struct mpn *mpn, int step, float alpha, float beta1, float beta2) {
    linear_adam(&mpn->W_i, step, alpha, beta1, beta2);
    linear_adam(&mpn->W_h, step, alpha, beta1, beta2);
    linear_adam(&mpn->W_o, step, alpha, beta1, beta2);
    for (int i = 0; i < FC_DEPTH; i++) {
        linear_adam(&mpn->fc[i], step, alpha, beta1, beta2);
    }
    linear_adam(&mpn->fco, step, alpha, beta1, beta2);
}
