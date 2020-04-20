#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mol.h"
#include "datapt.h"
#include "batch.h"
#include "cuda.h"
#include "kernels.h"

#define MP_DEPTH 3
#define FC_DEPTH 1
#define HIDDEN 300

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
    struct bias b_o; // hidden_size
    struct linear fc[FC_DEPTH]; // hidden_size x hidden_size
    struct bias fcb[FC_DEPTH]; // hidden_size
    struct linear fco; // hidden_size x 1
    struct bias fcob; // 1

    // activation storage
    // TODO surely there is a better way
    // TODO adopt more sustainable layer-oriented system

    // for the device-side mol
    struct mol d_mol;
    float *target;

    // for message passing
    struct act *mp_acts[MP_DEPTH + 1];  // message-passing activation storage
    float *mp_atoms[MP_DEPTH];          // outputs of atom-gather
    float *mp_bonds[MP_DEPTH];          // outputs of bond-scatter

    // for molecule embeddings
    float *out_atoms;
    float *out_atoms_f;
    struct act *out_act;
    float *embedding;

    // for final fully-connected layers
    struct act *fc_acts[FC_DEPTH];

    // for output
    float *finals;
};

// allocate memory for intermediate activations
void alloc_intermediate(struct mpn *mpn, int n_mols, int n_atoms, int n_bonds) {
    // allocate mol and target
    cuda_malloc((void **) &mpn->d_mol.f_atoms, sizeof(float) * n_atoms * ATOM_FDIM);
    cuda_malloc((void **) &mpn->d_mol.f_bonds, sizeof(float) * n_bonds * BOND_FDIM);
    cuda_malloc((void **) &mpn->d_mol.a_bonds, sizeof(int) * (n_atoms + 1));
    cuda_malloc((void **) &mpn->d_mol.a2b, sizeof(int) * n_bonds);
    cuda_malloc((void **) &mpn->d_mol.b2a, sizeof(int) * n_bonds);
    cuda_malloc((void **) &mpn->d_mol.b2revb, sizeof(int) * n_bonds);
    cuda_malloc((void **) &mpn->target, sizeof(float) * n_mols);

    // allocate activations
    for (int i = 0; i < MP_DEPTH + 1; i++) {
        mpn->mp_acts[i] = act_create(n_bonds * HIDDEN);
    }
    for (int i = 0; i < MP_DEPTH; i++) {
        cuda_malloc((void **) &mpn->mp_atoms[i], sizeof(float) * n_atoms * HIDDEN);
        cuda_malloc((void **) &mpn->mp_bonds[i], sizeof(float) * n_bonds * HIDDEN);
    }

    cuda_malloc((void **) &mpn->out_atoms, sizeof(float) * n_atoms * HIDDEN);
    cuda_malloc((void **) &mpn->out_atoms_f, sizeof(float) * n_atoms * (ATOM_FDIM + HIDDEN));
    mpn->out_act = act_create(n_atoms * HIDDEN);

    cuda_malloc((void **) &mpn->embedding, sizeof(float) * n_mols * HIDDEN);

    for (int i = 0; i < FC_DEPTH; i++) {
        mpn->fc_acts[i] = act_create(n_mols * HIDDEN);
    }

    cuda_malloc((void **) &mpn->finals, sizeof(float) * n_mols);
}

void free_intermediate(struct mpn *mpn) {
    // free mol and target
    cuda_free(mpn->d_mol.f_atoms);
    cuda_free(mpn->d_mol.f_bonds);
    cuda_free(mpn->d_mol.a_bonds);
    cuda_free(mpn->d_mol.a2b);
    cuda_free(mpn->d_mol.b2a);
    cuda_free(mpn->d_mol.b2revb);
    cuda_free(mpn->target);

    // free activations
    for (int i = 0; i < MP_DEPTH + 1; i++) {
        act_free(mpn->mp_acts[i]);
    }
    for (int i = 0; i < MP_DEPTH; i++) {
        cuda_free(mpn->mp_atoms[i]);
        cuda_free(mpn->mp_bonds[i]);
    }

    cuda_free(mpn->out_atoms);
    cuda_free(mpn->out_atoms_f);
    act_free(mpn->out_act);

    cuda_free(mpn->embedding);

    for (int i = 0; i < FC_DEPTH; i++) {
        act_free(mpn->fc_acts[i]);
    }

    cuda_free(mpn->finals);
}

struct mpn *mpn_create() {
    struct mpn *mpn = malloc(sizeof(struct mpn));

    linear_create(BOND_FDIM, HIDDEN, &mpn->W_i);
    linear_create(HIDDEN, HIDDEN, &mpn->W_h);
    linear_create(ATOM_FDIM + HIDDEN, HIDDEN, &mpn->W_o);
    bias_create(HIDDEN, &mpn->b_o);
    for (int i = 0; i < FC_DEPTH; i++) {
        linear_create(HIDDEN, HIDDEN, &mpn->fc[i]);
        bias_create(HIDDEN, &mpn->fcb[i]);
    }
    // TODO this is a vector dot and should not use sgemm
    linear_create(HIDDEN, 1, &mpn->fco);
    bias_create(1, &mpn->fcob);

    return mpn;
}

void mpn_init(struct mpn *mpn) {
    // XXX maximum batch size! 1000 mols, 5000 atoms, 10000 bonds
    alloc_intermediate(mpn, 1000, 5000, 10000);

    linear_init(&mpn->W_i);
    linear_init(&mpn->W_h);
    linear_init(&mpn->W_o);
    bias_init(&mpn->b_o);
    for (int i = 0; i < FC_DEPTH; i++) {
        linear_init(&mpn->fc[i]);
        bias_init(&mpn->fcb[i]);
    }
    linear_init(&mpn->fco);
    bias_init(&mpn->fcob);
}

void mol_to_device(struct mpn *mpn, struct batch *batch) {
    struct mol *mol = batch->mol;
    struct mol *d_mol = &mpn->d_mol;

    // copy molecule and target to device
    cuda_memcpy_htod(d_mol->f_atoms, mol->f_atoms, sizeof(float) * mol->n_atoms * ATOM_FDIM);
    cuda_memcpy_htod(d_mol->f_bonds, mol->f_bonds, sizeof(float) * mol->n_bonds * BOND_FDIM);
    cuda_memcpy_htod(d_mol->a_bonds, mol->a_bonds, sizeof(int) * (mol->n_atoms + 1));
    cuda_memcpy_htod(d_mol->a2b, mol->a2b, sizeof(int) * mol->n_bonds);
    cuda_memcpy_htod(d_mol->b2a, mol->b2a, sizeof(int) * mol->n_bonds);
    cuda_memcpy_htod(d_mol->b2revb, mol->b2revb, sizeof(int) * mol->n_bonds);
    cuda_memcpy_htod(mpn->target, batch->labels, sizeof(float) * batch->n_mols);
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

// message-passing network
float *mpn_forward(struct mpn *mpn, struct batch *batch) {
    struct mol *mol = batch->mol;
    int n_bonds = mol->n_bonds;
    int n_atoms = mol->n_atoms;
    int n_mols = batch->n_mols;

    mol_to_device(mpn, batch);

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
    bias_forward(&mpn->b_o, n_atoms, mpn->out_act->output);

    // average these to get the molecule embedding
    for (int i = 0; i < batch->n_mols; i++) {
        int start = batch->m_atoms[i];
        int end = batch->m_atoms[i + 1];
        average_forward(end - start, HIDDEN,
            mpn->out_act->output + start * HIDDEN,
            mpn->embedding + i * HIDDEN);
    }

    // one fully-connected feed-forward network, three layers
    // linear o activation o dropout
    // linear o activation to final value
    layer_forward(&mpn->fc[0], batch->n_mols, mpn->embedding, mpn->fc_acts[0]);
    bias_forward(&mpn->fcb[0], batch->n_mols, mpn->fc_acts[0]->output);
    for (int i = 1; i < FC_DEPTH; i++) {
        layer_forward(&mpn->fc[i], batch->n_mols, mpn->fc_acts[i - 1]->output, mpn->fc_acts[i]);
        bias_forward(&mpn->fcb[i], batch->n_mols, mpn->fc_acts[i]->output);
    }

    linear_forward(&mpn->fco, batch->n_mols, mpn->fc_acts[FC_DEPTH - 1]->output, mpn->finals);
    bias_forward(&mpn->fcob, batch->n_mols, mpn->finals);

    return mpn->finals;
}

float mpn_backward(struct mpn *mpn, struct batch *batch) {
    struct mol *mol = batch->mol;
    int n_bonds = mol->n_bonds;
    int n_atoms = mol->n_atoms;

    bceloss_backward(batch->n_mols, mpn->target, mpn->finals, mpn->finals);

    bias_backward(&mpn->fcob, batch->n_mols, mpn->finals);
    linear_backward(&mpn->fco, batch->n_mols, mpn->fc_acts[FC_DEPTH - 1]->output,
        mpn->finals, mpn->fc_acts[FC_DEPTH - 1]->output);

    for (int i = FC_DEPTH - 1; i > 0; i--) {
        bias_backward(&mpn->fcb[i], n_atoms, mpn->fc_acts[i]->output);
        layer_backward(&mpn->fc[i], batch->n_mols, mpn->fc_acts[i - 1]->output,
            mpn->fc_acts[i], mpn->fc_acts[i - 1]->output);
    }
    bias_backward(&mpn->fcb[0], batch->n_mols, mpn->fc_acts[0]->output);
    layer_backward(&mpn->fc[0], batch->n_mols, mpn->embedding, mpn->fc_acts[0], mpn->embedding);

    for (int i = 0; i < batch->n_mols; i++) {
        int start = batch->m_atoms[i];
        int end = batch->m_atoms[i + 1];
        average_backward(end - start, HIDDEN,
            mpn->embedding + i * HIDDEN,
            mpn->out_act->output + start * HIDDEN);
    }

    bias_backward(&mpn->b_o, n_atoms, mpn->out_act->output);
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

float mpn_loss(struct mpn *mpn, struct batch *batch) {
    float *losses;
    cuda_malloc((void **) &losses, sizeof(float) * batch->n_mols);

    bceloss_forward(batch->n_mols, mpn->target, mpn->finals, losses);

    // get the goods!!!
    float *h_losses = malloc(sizeof(float) * batch->n_mols);
    cuda_memcpy_dtoh(h_losses, losses, sizeof(float) * batch->n_mols);
    float loss = 0;
    for (int i = 0; i < batch->n_mols; i++) {
        loss += h_losses[i];
    }
    free(h_losses);
    cuda_free(losses);

    return loss;
}

float *mpn_test(struct mpn *mpn, struct batch *batch) {
    float *preds;
    cuda_malloc((void **) &preds, sizeof(float) * batch->n_mols);

    mpn_forward(mpn, batch);

    sigmoid_forward(batch->n_mols, mpn->finals, preds);

    float *h_preds = malloc(sizeof(float) * batch->n_mols);
    cuda_memcpy_dtoh(h_preds, preds, sizeof(float) * batch->n_mols);
    free_intermediate(mpn);

    return h_preds;
}
