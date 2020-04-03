#include <stdio.h>
#include <stdlib.h>

#include "mol.h"
#include "datapt.h"
#include "cuda.h"
#include "kernels.h"

#define MP_DEPTH 3
#define FC_DEPTH 2
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
    float *fco_out;
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

// allocate memory for intermediate activations
void mpn_alloc(struct mpn *mpn, struct mol *mol) { 
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
    cuda_malloc((void **) &mpn->fco_out, sizeof(float));
}

void mpn_free(struct mpn *mpn) {
    // TODO TODO TODO
}

// message-passing network
float mpn_forward(struct mpn *mpn, struct mol *mol) {
    int n_bonds = mol->n_bonds;
    int n_atoms = mol->n_atoms;

    mpn_alloc(mpn, mol);

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
    sigmoid_forward(1, mpn->fco_act, mpn->fco_out);

    // get the goods!!!
    float result;
    cuda_memcpy_dtoh(&result, mpn->fco_out, sizeof(float));

    return result;
}

float mpn_backward(struct mpn *mpn, struct mol *mol) {
    int n_bonds = mol->n_bonds;
    int n_atoms = mol->n_atoms;

    sigmoid_backward(1, mpn->fco_act, get_ones(1), mpn->fco_act);

    linear_backward(&mpn->fco, 1, mpn->fc_acts[FC_DEPTH - 1]->output,
        mpn->fco_act, mpn->fc_acts[FC_DEPTH - 1]->output);

    for (int i = FC_DEPTH - 1; i > 0; i--) {
        layer_backward(&mpn->fc[i], 1, mpn->fc_acts[i - 1]->output,
            mpn->fc_acts[i], mpn->fc_acts[i - 1]->output);
    }
    layer_backward(&mpn->fc[0], 1, mpn->embedding, mpn->fc_acts[0], mpn->embedding);

    // TODO use cublasSger instead for efficiency?
    average_backward(n_atoms, HIDDEN, mpn->embedding, mpn->out_act->output);

    layer_backward(&mpn->W_o, n_atoms, mpn->out_atoms_f,
        mpn->out_act, mpn->out_atoms_f);

    slice(n_atoms, ATOM_FDIM + HIDDEN, ATOM_FDIM, ATOM_FDIM + HIDDEN,
        mpn->out_atoms_f, mpn->out_atoms);

    atom_gather_backward(n_atoms, n_bonds, HIDDEN,
        mpn->d_mol.a_bonds, mpn->d_mol.a2b, mpn->mp_acts[MP_DEPTH]->output,
        mpn->out_atoms, mpn->mp_acts[MP_DEPTH]->output);

    float *message_grad;
    cuda_malloc((void **) &message_grad, sizeof(float) * n_bonds * HIDDEN);
    for (int i = MP_DEPTH - 1; i >= 0; i--) {
        layer_backward(&mpn->W_h, n_bonds, mpn->mp_bonds[i],
            mpn->mp_acts[i + 1], mpn->mp_bonds[i]);
        bond_scatter_backward(n_bonds, HIDDEN, mpn->d_mol.b2a, mpn->d_mol.b2revb,
            mpn->mp_atoms[i], mpn->mp_acts[i]->output, mpn->mp_bonds[i],
            mpn->mp_atoms[i], message_grad);
        atom_gather_backward(n_atoms, n_bonds, HIDDEN,
            mpn->d_mol.a_bonds, mpn->d_mol.a2b,
            mpn->mp_acts[i]->output, mpn->mp_atoms[i], mpn->mp_acts[i]->output);
        // TODO now add message_grad onto mpn->mp_acts[i]->output
    }

    // don't trash the input
    // TODO don't waste time here
    float *garbage;
    cuda_malloc((void **) &garbage, sizeof(float) * n_bonds * BOND_FDIM);
    layer_backward(&mpn->W_i, n_bonds, mpn->d_mol.f_bonds, mpn->mp_acts[0], garbage);

    mpn_free(mpn);
}
