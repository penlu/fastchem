#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mol.h"
#include "datapt.h"
#include "mpn.h"

void train(struct datapt *data, int N) {
    /*for (int epoch = 0; epoch < 10; epoch++) {
        // TODO data order randomization
        // TODO batch packing
    }*/
    struct mpn *mpn = mpn_create();

    mpn_init(mpn);

    for (int step = 1; step <= 10; step++) {
        float tot_loss = 0;
        for (int i = 0; i < N; i++) {
            struct mol *mol = data[i].mol;
            float target = data[i].label;
            if (mol->n_atoms && mol->n_bonds) {
                float loss = mpn_forward(mpn, mol, target);
                tot_loss += loss;

                mpn_backward(mpn, data[i].mol, target);
                mpn_adam(mpn, step, 0.001, 0.9, 0.999);
            }
        }
        printf("total loss %f\n", tot_loss);
    }
}

int main(int argc, char **argv) {
    int N;
    struct datapt *data = parse_datapts(stdin, 0, &N);

    printf("parsed %d data points\n", N);

    // run classification
    train(data, N);

    for (int i = 0; i < N; i++) {
        free_mol(data[i].mol);
    }
    free(data);

    return 0;
}
