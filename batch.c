#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mol.h"
#include "datapt.h"
#include "batch.h"

// create list of batches from list of molecules
// n: number of molecules
// l: list of n indices, for order randomization
// b: size of batch
// data: list of data points (molecule + label)
struct batch *batch_create(int n, int *l, int b, struct datapt *data) {
    // TODO make the batches
    int n_batches = n / b;
    struct batch *batches = malloc(sizeof(struct batch) * n_batches);

    for (int i = 0; i < n / b; i++) {
        batches[i].n_mols = b;
        batches[i].m_atoms = malloc(sizeof(int) * (b + 1));
        batches[i].mol = malloc(sizeof(struct mol));
        batches[i].labels = malloc(sizeof(float) * b);

        // count atoms and bonds
        int b_atoms = 0;
        int b_bonds = 0;
        for (int j = 0; j < b; j++) {
            b_atoms += data[l[i * b + j]].mol->n_atoms;
            b_bonds += data[l[i * b + j]].mol->n_bonds;
        }

        // initialize batch mol graph
        struct mol *bmol = batches[i].mol;
        bmol->n_atoms = b_atoms;
        bmol->n_bonds = b_bonds;
        bmol->f_atoms = malloc(sizeof(float) * b_atoms * ATOM_FDIM);
        bmol->f_bonds = malloc(sizeof(float) * b_bonds * BOND_FDIM);
        bmol->a_bonds = malloc(sizeof(int) * (b_atoms + 1));
        bmol->a2b = malloc(sizeof(int) * b_bonds);
        bmol->b2a = malloc(sizeof(int) * b_bonds);
        bmol->b2revb = malloc(sizeof(int) * b_bonds);

        // pack molecules
        int a_offset = 0;
        int b_offset = 0;
        for (int j = 0; j < b; j++) {
            struct mol *mol = data[l[i * b + j]].mol;
            int n_atoms = mol->n_atoms;
            int n_bonds = mol->n_bonds;

            batches[i].m_atoms[j] = a_offset;
            batches[i].labels[j] = data[l[i * b + j]].label;

            memcpy(bmol->f_atoms + a_offset * ATOM_FDIM, mol->f_atoms,
                sizeof(float) * n_atoms * ATOM_FDIM);
            memcpy(bmol->f_bonds + b_offset * BOND_FDIM, mol->f_bonds,
                sizeof(float) * n_bonds * BOND_FDIM);

            // copy bond indices
            for (int a = 0; a < n_atoms; a++) {
                bmol->a_bonds[a_offset + a] = mol->a_bonds[a] + b_offset;
            }
            for (int b = 0; b < n_bonds; b++) {
                bmol->a2b[b_offset + b] = mol->a2b[b] + b_offset;
                bmol->b2a[b_offset + b] = mol->b2a[b] + a_offset;
                bmol->b2revb[b_offset + b] = mol->b2revb[b] + b_offset;
            }

            a_offset += n_atoms;
            b_offset += n_bonds;
        }
        batches[i].m_atoms[b] = b_atoms;
        bmol->a_bonds[b_atoms] = b_bonds;
    }

    return batches;
}
