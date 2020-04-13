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

void test(struct mpn *, struct datapt *, int, int);

struct mpn *train(struct datapt *data, int N) {
    // prepare batches
    int l[N];
    for (int i = 0; i < N; i++) {
        l[i] = i;
    }

    shuffle(N, l);
    struct batch *batches = batch_create(N, l, BATCHSIZE, data);

    // create network
    struct mpn *mpn = mpn_create();

    mpn_init(mpn);

    // training loop
    int iters = 0;
    for (int epoch = 1; epoch <= 20; epoch++) {
        float tot_loss = 0;

        for (int i = 0; i < N / BATCHSIZE; i++) {
            mpn_forward(mpn, &batches[i]);

            float loss = mpn_loss(mpn, &batches[i]);
            tot_loss += loss;

            mpn_backward(mpn, &batches[i]);
            mpn_adam(mpn, ++iters, 0.001, 0.9, 0.999);
        }
        printf("epoch %d tot %f\n", epoch, tot_loss);
    }

    for (int i = 0; i < N / BATCHSIZE; i++) {
        free(batches[i].m_atoms);
        free_mol(batches[i].mol);
        free(batches[i].labels);
    }
    free(batches);

    return mpn;
}

void test(struct mpn *mpn, struct datapt *data, int start, int end) {
    int N = end - start;
    int l[N];
    for (int i = 0; i < N; i++) {
        l[i] = i;
    }

    struct batch *batch = batch_create(N, l, N, data + start);
    float *outputs = mpn_test(mpn, batch);

    int correct0 = 0;
    int correct1 = 0;
    int incorrect0 = 0;
    int incorrect1 = 0;
    int num0 = 0;
    int num1 = 0;
    for (int i = 0; i < N; i++) {
        if (batch->labels[i] == 1) {
            correct1 += (outputs[i] > 0.5);
            incorrect1 += (outputs[i] < 0.5);
            num1++;
        } else {
            correct0 += (outputs[i] < 0.5);
            incorrect0 += (outputs[i] > 0.5);
            num0++;
        }
    }
    free(outputs);

    free(batch->m_atoms);
    free_mol(batch->mol);
    free(batch->labels);
    free(batch);

    printf("%d correct of %d\n", correct0 + correct1, N);
    printf("%d of %d ones (%d)\n", correct1, num1, incorrect1);
    printf("%d of %d zeros (%d)\n", correct0, num0, incorrect0);
}

int main(int argc, char **argv) {
    int N;
    struct datapt *data = parse_datapts(stdin, 7, &N);

    printf("parsed %d data points\n", N);

    // run classification
    struct mpn *mpn = train(data, (int) (N * 0.8));
    test(mpn, data, (int) (N * 0.8), N);

    for (int i = 0; i < N; i++) {
        free_mol(data[i].mol);
    }
    free(data);

    return 0;
}
