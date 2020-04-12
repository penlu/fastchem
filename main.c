#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mol.h"
#include "datapt.h"
#include "batch.h"
#include "mpn.h"

void shuffle(int n, int *l) {
    for (int i = 0; i < n - 1; i++) {
        int j = (rand() % (n - i)) + i;
        int tmp = l[j];
        l[j] = l[i];
        l[i] = tmp;
    }
}

void train(struct datapt *data, int N) {
    struct mpn *mpn = mpn_create();

    mpn_init(mpn);

    int l[N];
    for (int i = 0; i < N; i++) {
        l[i] = i;
    }

    int iters = 0;
    for (int epoch = 1; epoch <= 1000; epoch++) {
        float tot_loss = 0;

        shuffle(N, l);
        struct batch *batches = batch_create(N, l, BATCHSIZE, data);

        for (int i = 0; i < N / BATCHSIZE; i++) {
            float loss = mpn_forward(mpn, &batches[i]);
            tot_loss += loss;
            //printf("    %d of %d %f\n", i, N/BATCHSIZE, loss);

            mpn_backward(mpn, &batches[i]);
            mpn_adam(mpn, ++iters, 0.001, 0.9, 0.999);
        }
        printf("epoch %d tot %f\n", epoch, tot_loss);

        for (int i = 0; i < N / BATCHSIZE; i++) {
            free(batches[i].m_atoms);
            free_mol(batches[i].mol);
            free(batches[i].labels);
        }
        free(batches);
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
