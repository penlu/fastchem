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
    for (int i = 0; i < N; i++) {
        if (data[i].mol->n_atoms && data[i].mol->n_bonds) {
            mpn_forward(mpn, data[i].mol);
            mpn_backward(mpn, data[i].mol);
        }
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
